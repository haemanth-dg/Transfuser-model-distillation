"""
StudentNet: lightweight autonomous driving policy for knowledge distillation.

Driving outputs only (no auxiliary heads):
    pred_checkpoint   [B, predict_checkpoint_len, 2]  — cumulative path checkpoints
    pred_target_speed [B, len(target_speeds)]          — speed-bin logits

KD feature signals (training only, not in ONNX graph):
    kd['bev']    [B, 64, 64, 64]   — BEV feature, matches teacher shape directly
    kd['fused']  [B, 256, 8, 8]    — fused spatial pooled to teacher's 8×8 spatial

    Teacher reference shapes (RegNetY-032, current config):
        teacher_bev   : [B, 64,  64, 64]  (bev_features_chanels=64)
        teacher_fused : [B, 256,  8,  8]  (change_channel output, reduction=32)

ONNX export:
    Use forward_inference() which returns only (pred_target_speed, pred_checkpoint).
    All KD projectors are gated by self.training and will not appear in the
    traced ONNX graph when the model is exported in eval() mode.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .student_backbone import StudentBackbone
from .model import GRUWaypointsPredictorInterFuser, PositionEmbeddingSine
from .center_net import LidarCenterNetHead
from . import transfuser_utils as t_u
from .lateral_controller import LateralPIDController
from .longitudinal_controller import LongitudinalLinearRegressionController


class StudentNet(nn.Module):
    """
    Parameters
    ----------
    config              : GlobalConfig — same object used for the teacher.
    use_kd_projectors   : bool — attach BEV + fused KD projection heads.
                          These are only active during training.
    """

    # Join hyper-params — matches teacher (config.num_transformer_decoder_layers=6,
    # config.num_decoder_heads=8)
    _JOIN_LAYERS = 6
    _JOIN_HEADS  = 8

    def __init__(self, config,
                 use_kd_projectors: bool = True):
        super().__init__()
        self.config            = config
        self.use_kd_projectors = use_kd_projectors

        # PID / longitudinal controllers (no parameters, inference only)
        self.lateral_pid_controller = LateralPIDController(config)
        self.longitudinal_controller = LongitudinalLinearRegressionController(config)

        # ---------------------------------------------------------------- backbone
        self.backbone = StudentBackbone(config)

        # ---------------------------------------------------------------- join
        # 1×1 conv: lidar_stage3_ch → gru_input_size  (same 256 as teacher)
        self.change_channel = nn.Conv2d(
            self.backbone.num_features, config.gru_input_size, kernel_size=1)

        self.encoder_pos_encoding = PositionEmbeddingSine(
            config.gru_input_size // 2, normalize=True)

        # Extra sensor encoder: [velocity(1), command(6)] → gru_input_size
        extra_size = 0
        if config.use_velocity:
            # BatchNorm1d fails at batch_size=1 in training; use Identity here
            # — the following Linear layers learn the scale instead.
            self.velocity_normalization = nn.Identity()
            extra_size += 1
        if config.use_discrete_command:
            extra_size += 6
        self.extra_sensor_encoder = nn.Sequential(
            nn.Linear(extra_size, 128), nn.ReLU(inplace=True),
            nn.Linear(128, config.gru_input_size), nn.ReLU(inplace=True))
        self.extra_sensor_pos_embed = nn.Parameter(
            torch.zeros(1, config.gru_input_size))

        # Target point token encoder
        target_point_size = 4 if config.two_tp_input else 2
        self._tp_size = target_point_size
        self.tp_encoder = nn.Sequential(
            nn.Linear(target_point_size, 128), nn.ReLU(inplace=True),
            nn.Linear(128, config.gru_input_size))
        self.tp_pos_embed = nn.Parameter(torch.zeros(1, config.gru_input_size))

        # Transformer decoder join — 2 layers vs teacher's 6
        decoder_norm  = nn.LayerNorm(config.gru_input_size)
        decoder_layer = nn.TransformerDecoderLayer(
            config.gru_input_size,
            self._JOIN_HEADS,
            activation=nn.GELU(),
            batch_first=True)
        self.join = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self._JOIN_LAYERS,
            norm=decoder_norm)

        # Learnable query tokens: predict_checkpoint_len path tokens + 1 speed token
        self.checkpoint_query = nn.Parameter(
            torch.zeros(1, config.predict_checkpoint_len + 1,
                        config.gru_input_size))

        # ---------------------------------------------------------------- decoders
        self.checkpoint_decoder = GRUWaypointsPredictorInterFuser(
            input_dim=config.gru_input_size,
            waypoints=config.predict_checkpoint_len,
            hidden_size=config.gru_hidden_size,
            target_point_size=target_point_size)

        self.target_speed_network = nn.Sequential(
            nn.Linear(config.gru_input_size, config.gru_input_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.gru_input_size, len(config.target_speeds)))
        
        #---------------------------------------------------------bounding box 
        # for training only, not used in student outputs
        # Also knowledge distillation is not done for this head
        if self.config.detect_boxes:
            self.head = LidarCenterNetHead(self.config)

        # ---------------------------------------------------------------- KD projectors
        # Hardcoded for current config (RegNetY-032 teacher):
        #   teacher_bev   : [B, 64, 64, 64]  — bev_features_chanels=64, spatial matches ✓
        #   teacher_fused : [B, 256, 8, 8]   — student is 256ch but 16×16, need pool
        if use_kd_projectors:
            # BEV: channel and spatial both match teacher → no-op
            self.bev_kd_proj = nn.Identity()

            # Fused: both 256ch but student spatial is 16×16, teacher is 8×8
            self.fused_kd_proj = nn.AdaptiveAvgPool2d((8, 8))

        # ---------------------------------------------------------------- init
        nn.init.uniform_(self.checkpoint_query)
        nn.init.uniform_(self.extra_sensor_pos_embed)
        nn.init.uniform_(self.tp_pos_embed)

    # ══════════════════════════════════════════════════════════════════════════
    # Forward
    # ══════════════════════════════════════════════════════════════════════════

    def forward(self, rgb, lidar_bev, target_point, ego_vel, command,
                target_point_next=None, timer=None, bbox_only: bool = False):
        """
        Returns a tuple compatible with the teacher's 10-element output, plus a
        KD feature dict as the 11th element.

        Index mapping (matches LidarCenterNet.forward):
            0  pred_wp            → None  (student has no separate WP head)
            1  pred_target_speed  → [B, n_speed_bins]
            2  pred_checkpoint    → [B, predict_checkpoint_len, 2]
            3  pred_semantic      → None
            4  pred_bev_semantic  → None
            5  pred_depth         → None
            6  pred_bounding_box  → None
            7  attention_weights  → None
            8  pred_wp_1          → None
            9  selected_path      → None
           10  kd_features        → dict (non-empty only when self.training)
        """
        _t = timer
        bs = rgb.shape[0]

        if self.config.two_tp_input and not bbox_only:
            target_point = torch.cat((target_point, target_point_next), dim=1)

        # ── backbone ────────────────────────────────────────────────────────
        if _t:
            with _t.measure('student/backbone_TOTAL'):
                bev_feat, fused_feat, img_feat = self.backbone(
                    rgb, lidar_bev, timer=_t)
        else:
            bev_feat, fused_feat, img_feat = self.backbone(rgb, lidar_bev)

        # ── channel proj + positional encoding ──────────────────────────────
        if _t:
            with _t.measure('student/join/channel_proj_posenc'):
                fused_spatial = self.change_channel(fused_feat)     # [B, 256, h, w]
                fused = fused_spatial + self.encoder_pos_encoding(fused_spatial)
                fused = torch.flatten(fused, start_dim=2)           # [B, 256, T]
        else:
            fused_spatial = self.change_channel(fused_feat)
            fused = fused_spatial + self.encoder_pos_encoding(fused_spatial)
            fused = torch.flatten(fused, start_dim=2)

        if bbox_only:
            # Skip planning head compute entirely; only bbox head + KD features.
            pred_checkpoint = None
            pred_target_speed = None
        else:
            # ── extra sensor tokens ─────────────────────────────────────────────
            if _t:
                with _t.measure('student/join/extra_sensor_encoder'):
                    sensors = []
                    if self.config.use_velocity:
                        sensors.append(self.velocity_normalization(ego_vel))
                    if self.config.use_discrete_command:
                        sensors.append(command)
                    sensors = self.extra_sensor_encoder(torch.cat(sensors, dim=1))
            else:
                sensors = []
                if self.config.use_velocity:
                    sensors.append(self.velocity_normalization(ego_vel))
                if self.config.use_discrete_command:
                    sensors.append(command)
                sensors = self.extra_sensor_encoder(torch.cat(sensors, dim=1))

            # Append extra-sensor token and target-point token to memory sequence
            sensors = sensors + self.extra_sensor_pos_embed.expand(bs, -1)
            fused = torch.cat((fused, sensors.unsqueeze(2)), dim=2)     # [B, 256, T+1]

            tp_token = self.tp_encoder(target_point) + self.tp_pos_embed.expand(bs, -1)
            fused = torch.cat((fused, tp_token.unsqueeze(2)), dim=2)    # [B, 256, T+2]

            fused = fused.permute(0, 2, 1)                              # [B, T+2, 256]

            # ── transformer decoder ─────────────────────────────────────────────
            if _t:
                with _t.measure('student/join/transformer_decoder_checkpoint'):
                    joined = self.join(
                        self.checkpoint_query.expand(bs, -1, -1), fused)
            else:
                joined = self.join(
                    self.checkpoint_query.expand(bs, -1, -1), fused)

            gru_feat = joined[:, :self.config.predict_checkpoint_len]   # [B, 10, 256]
            ts_feat  = joined[:, self.config.predict_checkpoint_len]    # [B, 256]

            # ── decoders ────────────────────────────────────────────────────────
            if _t:
                with _t.measure('student/decoder/gru_checkpoint'):
                    pred_checkpoint = self.checkpoint_decoder(gru_feat, target_point)
                with _t.measure('student/decoder/target_speed_network'):
                    pred_target_speed = self.target_speed_network(ts_feat)
            else:
                pred_checkpoint   = self.checkpoint_decoder(gru_feat, target_point)
                pred_target_speed = self.target_speed_network(ts_feat)

        # ── bounding box head (training only, not in student outputs) ─────────
        pred_bounding_box = None
        if self.config.detect_boxes:
            if _t:
                with _t.measure('student/head/centernet_detection'):
                    pred_bounding_box = self.head(bev_feat)
            else:
                pred_bounding_box = self.head(bev_feat)

        # ── KD features (training only — not traced into ONNX graph) ────────
        kd = {}
        if self.training and self.use_kd_projectors:
            kd['bev']   = self.bev_kd_proj(bev_feat)        # [B, 64, 64, 64]
            kd['fused'] = self.fused_kd_proj(fused_spatial)  # [B, 256, 8, 8]

        return (None, pred_target_speed, pred_checkpoint,
                None, None, None, pred_bounding_box, None, None, None,
                kd)

    # ══════════════════════════════════════════════════════════════════════════
    # Inference-only path (ONNX export target)
    # ══════════════════════════════════════════════════════════════════════════

    def forward_inference(self, rgb, lidar_bev, target_point, ego_vel, command):
        """
        Clean two-output forward pass for ONNX export.
        Returns: (pred_target_speed [B, n_bins], pred_checkpoint [B, 10, 2])

        Export example:
            model.eval()
            torch.onnx.export(
                model,
                (rgb, lidar, tp, vel, cmd),
                'student.onnx',
                opset_version=17,
                input_names=['rgb', 'lidar', 'tp', 'vel', 'cmd'],
                output_names=['target_speed', 'checkpoints'],
            )
        """
        out = self.forward(rgb, lidar_bev, target_point, ego_vel, command)
        return out[1], out[2]

    # ══════════════════════════════════════════════════════════════════════════
    # PID control helpers (same logic as teacher — no NN params)
    # ══════════════════════════════════════════════════════════════════════════

    def control_pid_direct(self, pred_checkpoints, pred_target_speed, speed,
                           ego_vehicle_location=0, ego_vehicle_rotation=0):
        if isinstance(speed, torch.Tensor):
            speed = speed[0].item()
        elif isinstance(speed, np.ndarray):
            speed = float(speed[0])
        if isinstance(pred_target_speed, torch.Tensor):
            pred_target_speed = pred_target_speed.item()

        brake = (pred_target_speed < 0.01
                 or (speed / pred_target_speed) > self.config.brake_ratio)

        steer = self.lateral_pid_controller.step(
            pred_checkpoints, speed,
            ego_vehicle_location, ego_vehicle_rotation,
            inference_mode=True)

        throttle, control_brake = self.longitudinal_controller.get_throttle_and_brake(
            brake, pred_target_speed, speed)

        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        steer    = np.clip(round(float(steer), 3), -1.0, 1.0)
        return steer, throttle, control_brake

    def control_pid(self, waypoints, velocity, tuned_aim_distance=False):
        assert waypoints.size(0) == 1
        waypoints = waypoints[0].data.cpu().numpy()
        speed = velocity[0].data.cpu().numpy()

        one_second  = int(self.config.carla_fps
                          // (self.config.wp_dilation * self.config.data_save_freq))
        half_second = one_second // 2
        desired_speed = np.linalg.norm(
            waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0

        brake = (desired_speed < self.config.brake_speed
                 or (speed / desired_speed) > self.config.brake_ratio)

        delta    = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = np.clip(self.speed_controller.step(delta), 0.0,
                           self.config.clip_throttle)
        throttle = throttle if not brake else 0.0

        aim_distance = (self.config.aim_distance_slow
                        if desired_speed < self.config.aim_distance_threshold
                        else self.config.aim_distance_fast)

        aim_index = waypoints.shape[0] - 1
        for idx, wp in enumerate(waypoints):
            if np.linalg.norm(wp) >= aim_distance:
                aim_index = idx
                break

        aim   = waypoints[aim_index]
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01 or brake:
            angle = 0.0

        steer = np.clip(self.turn_controller.step(angle), -1.0, 1.0)
        return steer, throttle, brake
