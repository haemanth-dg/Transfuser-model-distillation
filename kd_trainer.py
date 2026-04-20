"""
KD training loop: StudentNet distilled from LidarCenterNet (teacher).

Hardcoded for current config (all_towns, RegNetY-032 teacher):
    rgb       : [B, 3, 384, 1024]
    lidar_bev : [B, 1, 256, 256]
    target_point : [B, 2]
    ego_vel   : [B, 1]
    command   : [B, 6]

Teacher internal shapes captured via hooks:
    bev_feat  : [B, 64,  64, 64]
    fused     : [B, 256,  8,  8]   (after teacher.change_channel)

Student KD outputs (training only):
    kd['bev']   : [B, 64,  64, 64]   matches teacher ✓
    kd['fused'] : [B, 256,  8,  8]   pooled to match teacher ✓

Total loss:
    L = w_ck * L_checkpoint
      + w_ts * L_target_speed_task
      + w_feat * (L_bev_kd + L_fused_kd)
    + w_kd * L_speed_kd
    + w_bbox * L_bbox

Expected batch dict keys:
    rgb, lidar_bev, target_point, ego_vel, command  — sensor inputs
    checkpoints   : [B, 10, 2]   GT path checkpoints
    target_speed  : [B]  long    GT speed-bin class index (0–7)
    center_heatmap_label, wh_label, yaw_class_label, yaw_res_label,
    offset_label, velocity_label, brake_target_label, pixel_weight_label,
    avg_factor_label — CenterNet supervision targets
""" 

import os
import json
import re
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image

from .config import GlobalConfig
from .model import LidarCenterNet
from .student_model import StudentNet


def _infer_epoch_from_path(path):
    name = os.path.basename(str(path))
    match = re.search(r"epoch(\d+)", name)
    if match is None:
        match = re.search(r"model_(\d+)\.pth$", name)
    if match is None:
        return None
    return int(match.group(1))


def _state_paths(output_dir, epoch=None):
    model_name = f'model_{int(epoch):04d}.pth' if epoch is not None else None
    return {
        'model': os.path.join(output_dir, model_name) if model_name is not None else None,
        'best_model': os.path.join(output_dir, 'best_model.pth'),
        'optimizer': os.path.join(output_dir, 'optimizer.pth'),
        'scheduler': os.path.join(output_dir, 'scheduler.pth'),
        'scaler': os.path.join(output_dir, 'scaler.pth'),
        'meta_json': os.path.join(output_dir, 'meta_data.json'),
    }


def _save_epoch_state(
    output_dir,
    epoch,
    student,
    optimizer,
    scheduler,
    scaler,
    best_val_total,
    hyperparameters=None,
):
    os.makedirs(output_dir, exist_ok=True)
    paths = _state_paths(output_dir, epoch=epoch)

    torch.save(student.state_dict(), paths['model'])
    torch.save(optimizer.state_dict(), paths['optimizer'])
    torch.save(scheduler.state_dict(), paths['scheduler'])
    torch.save(
        {
            'enabled': bool(scaler is not None),
            'state_dict': scaler.state_dict() if scaler is not None else None,
        },
        paths['scaler'],
    )

    meta = {
        'epoch': int(epoch),
        'best_val_total': float(best_val_total),
        'latest_model_file': os.path.basename(paths['model']),
        'optimizer_file': os.path.basename(paths['optimizer']),
        'scheduler_file': os.path.basename(paths['scheduler']),
        'scaler_file': os.path.basename(paths['scaler']),
        'hyperparameters': hyperparameters or {},
    }
    with open(paths['meta_json'], 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


def _load_resume_state(loadfile, device):
    if not os.path.isfile(loadfile):
        raise FileNotFoundError(loadfile)

    resume_dir = os.path.dirname(loadfile) or '.'
    paths = _state_paths(resume_dir)
    checkpoint = {
        'student_state_dict': torch.load(loadfile, map_location=device),
        'optimizer_state_dict': torch.load(paths['optimizer'], map_location=device),
        'scheduler_state_dict': torch.load(paths['scheduler'], map_location=device),
        'scaler_state_dict': None,
        'epoch': None,
        'best_val_total': None,
    }

    if os.path.isfile(paths['scaler']):
        scaler_payload = torch.load(paths['scaler'], map_location=device)
        if isinstance(scaler_payload, dict) and 'state_dict' in scaler_payload:
            checkpoint['scaler_state_dict'] = scaler_payload.get('state_dict', None)
        else:
            checkpoint['scaler_state_dict'] = scaler_payload

    if os.path.isfile(paths['meta_json']):
        with open(paths['meta_json'], 'r', encoding='utf-8') as f:
            meta = json.load(f)
        checkpoint['epoch'] = meta.get('epoch', None)
        checkpoint['best_val_total'] = meta.get('best_val_total', None)

    return checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Teacher feature hook
# ──────────────────────────────────────────────────────────────────────────────

class _TeacherHooks:
    """
    Captures teacher's BEV and fused-spatial features without modifying
    the teacher model.  Register once; read .bev / .fused after each
    teacher forward call.
    """
    def __init__(self, teacher: LidarCenterNet):
        self.bev   = None
        self.fused = None
        self._handles = [
            teacher.backbone.register_forward_hook(self._hook_backbone),
            teacher.change_channel.register_forward_hook(self._hook_cc),
        ]

    def _hook_backbone(self, _m, _inp, out):
        # backbone returns (bev_feature_grid, fused_raw, image_feature_grid)
        self.bev = out[0]           # [B, 64, 64, 64]

    def _hook_cc(self, _m, _inp, out):
        self.fused = out            # [B, 256, 8, 8]

    def remove(self):
        for h in self._handles:
            h.remove()


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def compute_kd_loss(
    student: StudentNet,
    student_out,
    teacher_out,
    teacher_hooks: _TeacherHooks,
    batch,
    device,
    w_ck:   float = 1.0,
    w_ts:   float = 1.0,
    w_feat: float = 0.5,
    w_kd:   float = 0.5,
    w_bbox: float = 1.0,
    T:      float = 4.0,
):
    """
    Returns total scalar loss and a dict of component losses for logging.

    student_out : 11-tuple from StudentNet.forward()
    teacher_out : 10-tuple from LidarCenterNet.forward()
    """
    pred_target_speed = student_out[1]   # [B, 8]
    pred_checkpoint   = student_out[2]   # [B, 10, 2]
    pred_bounding_box = student_out[6]   # tuple of bbox head outputs
    kd                = student_out[10]  # dict with 'bev', 'fused'

    gt_checkpoints  = batch['checkpoints'].to(device)    # [B, 10, 2]
    gt_speed_cls    = batch['target_speed'].to(device)   # [B] long

    # ── task losses ─────────────────────────────────────────────────────────
    L_ck = F.smooth_l1_loss(pred_checkpoint, gt_checkpoints)
    L_ts = F.cross_entropy(pred_target_speed, gt_speed_cls)

    # ── feature KD losses ────────────────────────────────────────────────────
    # teacher features captured by hooks during teacher.forward()
    teacher_bev   = teacher_hooks.bev.detach()    # [B, 64, 64, 64]
    teacher_fused = teacher_hooks.fused.detach()  # [B, 256, 8, 8]

    L_bev_kd   = F.mse_loss(kd['bev'],   teacher_bev)
    L_fused_kd = F.mse_loss(kd['fused'], teacher_fused)

    # ── output KD: soft speed targets ────────────────────────────────────────
    teacher_speed = teacher_out[1].detach()   # [B, 8]
    L_speed_kd = F.kl_div(
        F.log_softmax(pred_target_speed / T, dim=-1),
        F.softmax(teacher_speed / T, dim=-1),
        reduction='batchmean',
    ) * (T ** 2)

    # ── supervised bbox loss ────────────────────────────────────────────────
    bbox_losses = {}
    L_bbox = torch.zeros((), device=device)
    if getattr(student.config, 'detect_boxes', False):
        if pred_bounding_box is None:
            raise RuntimeError('Student bbox head is enabled but student_out[6] is None.')
        center_heatmap_label = batch['center_heatmap_label'].to(device)
        wh_label = batch['wh_label'].to(device)
        yaw_class_label = batch['yaw_class_label'].to(device)
        yaw_res_label = batch['yaw_res_label'].to(device)
        offset_label = batch['offset_label'].to(device)
        velocity_label = batch['velocity_label'].to(device)
        brake_target_label = batch['brake_target_label'].to(device)
        pixel_weight_label = batch['pixel_weight_label'].to(device)
        avg_factor_label = batch['avg_factor_label'].to(device)

        bbox_losses = student.head.loss(
            pred_bounding_box[0], pred_bounding_box[1], pred_bounding_box[2], pred_bounding_box[3],
            pred_bounding_box[4], pred_bounding_box[5], pred_bounding_box[6],
            center_heatmap_label, wh_label, yaw_class_label, yaw_res_label, offset_label,
            velocity_label, brake_target_label, pixel_weight_label, avg_factor_label,
        )
        L_bbox = torch.stack([v for v in bbox_losses.values()]).sum()

    # ── total ─────────────────────────────────────────────────────────────────
    total = (w_ck   * L_ck
           + w_ts   * L_ts
           + w_feat * (L_bev_kd + L_fused_kd)
           + w_kd   * L_speed_kd
           + w_bbox * L_bbox)

    components = {
        'checkpoint':  L_ck.item(),
        'target_speed_task': L_ts.item(),
        'bev_kd':      L_bev_kd.item(),
        'fused_kd':    L_fused_kd.item(),
        'speed_kd':    L_speed_kd.item(),
        'bbox_total':  L_bbox.item(),
        'total':       total.item(),
    }
    for name, loss_val in bbox_losses.items():
        components[name] = loss_val.item()

    return total, components


# ──────────────────────────────────────────────────────────────────────────────
# Train / val loops
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    teacher:  LidarCenterNet,
    student:  StudentNet,
    loader:   DataLoader,
    optimizer: torch.optim.Optimizer,
    device:   torch.device,
    hooks:    _TeacherHooks,
    loss_weights: dict,
    scaler:   torch.cuda.amp.GradScaler = None,
    log_every: int = 50,
):
    student.train()
    teacher.eval()

    running = {}
    n = 0

    for step, batch in tqdm(enumerate(loader), total=len(loader)):
        rgb         = batch['rgb'].to(device)
        lidar_bev   = batch['lidar_bev'].to(device)
        target_point= batch['target_point'].to(device)
        ego_vel     = batch['ego_vel'].to(device)
        command     = batch['command'].to(device)

        # print(f"rgb shape: {rgb[0].shape}, lidar_bev shape: {lidar_bev[0].shape}")
        # pil_rgb = Image.fromarray((rgb[0].cpu().permute(1, 2, 0).numpy()).astype('uint8'))
        # pil_rgb.save('debug_rgb.png')
        # pil_lidar_bev = Image.fromarray((lidar_bev[0].cpu().numpy()[0] * 255).astype('uint8'))
        # pil_lidar_bev.save('debug_lidar_bev.png')
        # print('Saved debug_rgb.png and debug_lidar_bev.png for inspection.')

        # ── teacher forward (no grad, eval BN) ───────────────────────────────
        with torch.no_grad():
            teacher_out = teacher(rgb, lidar_bev, target_point, ego_vel, command)

        # ── student forward ───────────────────────────────────────────────────
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type='cuda'):
                student_out = student(rgb, lidar_bev, target_point, ego_vel, command)
                loss, comps = compute_kd_loss(
                    student, student_out, teacher_out, hooks, batch, device,
                    **loss_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer) 
            scaler.update()
        else:
            student_out = student(rgb, lidar_bev, target_point, ego_vel, command)
            loss, comps = compute_kd_loss(
                student, student_out, teacher_out, hooks, batch, device,
                **loss_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

        for k, v in comps.items():
            running[k] = running.get(k, 0.0) + v
        n += 1

        if log_every > 0 and (step + 1) % log_every == 0:
            avg = {k: running[k] / n for k in running}
            parts = ' | '.join(f'{k}={v:.4f}' for k, v in avg.items())
            print(f'  step {step+1}/{len(loader)}  {parts}')

    return {k: running[k] / max(n, 1) for k in running}


@torch.no_grad()
def validate(
    teacher:  LidarCenterNet,
    student:  StudentNet,
    loader:   DataLoader,
    device:   torch.device,
    hooks:    _TeacherHooks,
    loss_weights: dict,
):
    student.eval()
    teacher.eval()

    running = {}
    n = 0

    for batch in tqdm(loader):
        rgb         = batch['rgb'].to(device)
        lidar_bev   = batch['lidar_bev'].to(device)
        target_point= batch['target_point'].to(device)
        ego_vel     = batch['ego_vel'].to(device)
        command     = batch['command'].to(device)

        teacher_out = teacher(rgb, lidar_bev, target_point, ego_vel, command)

        # student in eval: kd dict is empty — run hooks-only path for val loss
        # temporarily switch to train so kd projectors fire
        student.train()
        student_out = student(rgb, lidar_bev, target_point, ego_vel, command)
        student.eval()

        _, comps = compute_kd_loss(
            student, student_out, teacher_out, hooks, batch, device, **loss_weights)

        for k, v in comps.items():
            running[k] = running.get(k, 0.0) + v
        n += 1

    return {k: running[k] / max(n, 1) for k in running}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def train(
    config_path:  str,
    teacher_ckpt: str,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    output_dir:   str,
    epochs:       int = 40,
    lr:           float = 3e-4,
    weight_decay: float = 1e-4,
    use_amp:      bool = True,
    save_every:   int = 5,
    log_every:    int = 50,
    loss_weights: dict = None,
    detect_boxes: bool = True,
    loadfile:     str = None,
    continue_epoch: int = 0,
    hyperparameters: dict = None,
):
    """
    Main KD training entry point.

    Args:
        config_path  : path to config.json (same one used for teacher)
        teacher_ckpt : path to teacher .pth checkpoint
        train_loader : DataLoader yielding dicts with keys:
                         rgb [B,3,384,1024], lidar_bev [B,1,256,256],
                         target_point [B,2], ego_vel [B,1], command [B,6],
                         checkpoints [B,10,2], target_speed [B] (long)
        val_loader   : same format as train_loader
        output_dir   : directory to save student checkpoints + logs
        loss_weights : dict with optional keys w_ck, w_ts, w_feat, w_kd, T
                       (defaults applied for any missing key)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('Warning: CUDA not available. Running KD training on CPU.')
    os.makedirs(output_dir, exist_ok=True)

    # ── config ────────────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg_dict = json.load(f)
    cfg_dict.pop('setting', None)
    config = GlobalConfig()
    config.initialize(setting='eval', **cfg_dict)
    config.compile = False
    config.detect_boxes = bool(detect_boxes)

    # ── teacher (frozen) ──────────────────────────────────────────────────────
    teacher = LidarCenterNet(config)
    state = torch.load(teacher_ckpt, map_location='cpu')
    teacher.load_state_dict(state, strict=True)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    hooks = _TeacherHooks(teacher)

    # ── student ───────────────────────────────────────────────────────────────
    student = StudentNet(config, use_kd_projectors=True).to(device)
    student_param_count = sum(p.numel() for p in student.parameters())
    print(f'Student parameter count: {student_param_count:,}')

    if config.detect_boxes and loadfile is None:
        if not hasattr(teacher, 'head') or not hasattr(student, 'head'):
            raise RuntimeError('detect_boxes=True but teacher/student bbox head is unavailable.')
        student.head.load_state_dict(teacher.head.state_dict(), strict=True)
        print('Initialized student bbox head from teacher checkpoint weights.')

    start_epoch = 1
    best_val_total = float('inf')

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    tb_dir = os.path.join(output_dir)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f'TensorBoard logs: {tb_dir}')

    def _assert_resume_has_compatible_bbox_head(state_dict):
        if not config.detect_boxes:
            return
        if not hasattr(student, 'head'):
            raise RuntimeError('detect_boxes=True but student model has no bbox head.')

        expected = student.head.state_dict()
        possible_prefixes = ('head.', 'module.head.')
        missing = []
        mismatched = []

        for key, value in expected.items():
            loaded_value = None
            for prefix in possible_prefixes:
                full_key = f'{prefix}{key}'
                if full_key in state_dict:
                    loaded_value = state_dict[full_key]
                    break
            if loaded_value is None:
                missing.append(key)
                continue
            if tuple(loaded_value.shape) != tuple(value.shape):
                mismatched.append((key, tuple(value.shape), tuple(loaded_value.shape)))

        if missing or mismatched:
            details = []
            if missing:
                details.append(f'missing keys: {missing[:5]}')
            if mismatched:
                preview = [f'{k} expected {es} got {gs}' for k, es, gs in mismatched[:3]]
                details.append('shape mismatches: ' + '; '.join(preview))
            raise RuntimeError(
                'Resume checkpoint is incompatible with bbox-head training. '
                + ' | '.join(details)
            )

    if loadfile is not None:
        if not os.path.isfile(loadfile):
            raise FileNotFoundError(loadfile)
        print(f'Loading student resume file: {loadfile}')
        checkpoint = _load_resume_state(loadfile, device)
        inferred_epoch = _infer_epoch_from_path(loadfile)

        def _resolve_resume_epoch(checkpoint_epoch):
            candidates = []
            if checkpoint_epoch is not None:
                candidates.append(int(checkpoint_epoch))
            if inferred_epoch is not None:
                candidates.append(int(inferred_epoch))
            if continue_epoch is not None and int(continue_epoch) > 0:
                candidates.append(int(continue_epoch))
            return max(candidates) if candidates else 0

        _assert_resume_has_compatible_bbox_head(checkpoint['student_state_dict'])
        student.load_state_dict(checkpoint['student_state_dict'], strict=True)
        if checkpoint.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        best_val_total = float(checkpoint.get('best_val_total', best_val_total))
        resumed_epoch = _resolve_resume_epoch(checkpoint.get('epoch', None))
        start_epoch = resumed_epoch + 1
        print(f'Resume epoch resolved from metadata/file/arg: {resumed_epoch}')
        print(f'Resuming from epoch {start_epoch}')

    # ── loss weights ──────────────────────────────────────────────────────────
    _defaults = dict(w_ck=1.0, w_ts=1.0, w_feat=0.5, w_kd=0.5, T=4.0)
    lw = {**_defaults, **(loss_weights or {})}
    hp = dict(hyperparameters or {})
    hp.setdefault('epochs', int(epochs))
    hp.setdefault('lr', float(lr))
    hp.setdefault('weight_decay', float(weight_decay))
    hp.setdefault('use_amp', bool(use_amp))
    hp.setdefault('save_every', int(save_every))
    hp.setdefault('log_every', int(log_every))
    hp.setdefault('detect_boxes', bool(detect_boxes))
    hp.setdefault('continue_epoch', int(continue_epoch))
    hp.setdefault('loss_weights', dict(lw))

    # ── training loop ─────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, epochs + 1):
            print(f'\n[Epoch {epoch}/{epochs}]  lr={scheduler.get_last_lr()[0]:.2e}')

            train_metrics = train_one_epoch(
                teacher, student, train_loader, optimizer, device,
                hooks, lw, scaler=scaler, log_every=log_every)
            scheduler.step()

            val_metrics = validate(
                teacher, student, val_loader, device, hooks, lw)

            # log
            print(f'  TRAIN  ' +
                  '  '.join(f'{k}={v:.4f}' for k, v in train_metrics.items()))
            print(f'  VAL    ' +
                  '  '.join(f'{k}={v:.4f}' for k, v in val_metrics.items()))

            for key, value in train_metrics.items():
                writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f'val/{key}', value, epoch)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
            writer.flush()

            # save best
            if val_metrics['total'] < best_val_total:
                best_val_total = val_metrics['total']
                torch.save(student.state_dict(), _state_paths(output_dir)['best_model'])
                print(f'  ✓ saved best  (val_total={best_val_total:.4f})')

            _save_epoch_state(
                output_dir,
                epoch,
                student,
                optimizer,
                scheduler,
                scaler,
                best_val_total,
                hyperparameters=hp,
            )
    finally:
        writer.close()
        hooks.remove()

    print('\nTraining complete.')
    return student
