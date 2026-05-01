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


def _load_finetuning_state(loadfile, device):
    """Load only model weights for fine-tuning (continue_epoch=0 mode)."""
    if not os.path.isfile(loadfile):
        raise FileNotFoundError(loadfile)
    student_state_dict = torch.load(loadfile, map_location=device)
    return student_state_dict


def _load_resume_state(loadfile, device):
    """Load full checkpoint state for resume training (continue_epoch>0 mode)."""
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
        'hyperparameters': {},
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
        checkpoint['hyperparameters'] = meta.get('hyperparameters', {})
    else:
        raise FileNotFoundError(
            f'Resume mode requires meta_data.json at {paths["meta_json"]}'
        )

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


def _get_bbox_keys_from_state_dict(state_dict):
    """Extract all head.* keys from state_dict."""
    head_keys = set()
    for key in state_dict.keys():
        if key.startswith('head.'):
            head_keys.add(key)
        elif key.startswith('module.head.'):
            head_keys.add(key)
    return head_keys


def _check_bbox_key_presence(state_dict):
    """
    Check bbox key presence in state_dict.
    Returns: 'present' or 'absent'.
    """
    head_keys = _get_bbox_keys_from_state_dict(state_dict)
    return 'present' if head_keys else 'absent'


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
    bbox_only: bool = False,
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
    pred_target_speed = student_out[1]   # [B, 8] or None in bbox-only
    pred_checkpoint   = student_out[2]   # [B, 10, 2] or None in bbox-only
    pred_bounding_box = student_out[6]   # tuple of bbox head outputs
    kd                = student_out[10]  # dict with 'bev', 'fused'

    # ── task losses (planning) ─────────────────────────────────────────────
    if bbox_only:
        L_ck = torch.zeros((), device=device)
        L_ts = torch.zeros((), device=device)
    else:
        gt_checkpoints  = batch['checkpoints'].to(device)    # [B, 10, 2]
        gt_speed_cls    = batch['target_speed'].to(device)   # [B] long
        L_ck = F.smooth_l1_loss(pred_checkpoint, gt_checkpoints)
        L_ts = F.cross_entropy(pred_target_speed, gt_speed_cls)

    # ── feature KD losses ────────────────────────────────────────────────────
    # teacher features captured by hooks during teacher.forward()
    teacher_bev   = teacher_hooks.bev.detach()    # [B, 64, 64, 64]
    teacher_fused = teacher_hooks.fused.detach()  # [B, 256, 8, 8]

    L_bev_kd   = F.mse_loss(kd['bev'],   teacher_bev)
    L_fused_kd = F.mse_loss(kd['fused'], teacher_fused)

    # ── output KD: soft speed targets ────────────────────────────────────────
    if bbox_only:
        L_speed_kd = torch.zeros((), device=device)
    else:
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


def _make_dummy_planning_inputs(config: GlobalConfig, batch_size: int, device: torch.device):
    tp_size = 4 if getattr(config, 'two_tp_input', False) else 2
    target_point = torch.zeros((batch_size, tp_size), device=device)
    target_point_next = torch.zeros((batch_size, tp_size), device=device)
    ego_vel = torch.zeros((batch_size, 1), device=device)
    command = torch.zeros((batch_size, 6), device=device)
    return target_point, target_point_next, ego_vel, command


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
    bbox_only: bool = False,
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

        if bbox_only:
            target_point, target_point_next, ego_vel, command = _make_dummy_planning_inputs(
                student.config, rgb.shape[0], device)
        else:
            target_point= batch['target_point'].to(device)
            target_point_next = None
            ego_vel     = batch['ego_vel'].to(device)
            command     = batch['command'].to(device)

        print(f"rgb shape: {rgb[0].shape}, lidar_bev shape: {lidar_bev[0].shape}")
        pil_rgb = Image.fromarray((rgb[0].cpu().permute(1, 2, 0).numpy()).astype('uint8'))
        pil_rgb.save('debug_rgb.png')
        pil_lidar_bev = Image.fromarray((lidar_bev[0].cpu().numpy()[0] * 255).astype('uint8'))
        pil_lidar_bev.save('debug_lidar_bev.png')
        print('Saved debug_rgb.png and debug_lidar_bev.png for inspection.')

        # ── teacher forward (no grad, eval BN) ───────────────────────────────
        with torch.no_grad():
            teacher_out = teacher(
                rgb, lidar_bev, target_point, ego_vel, command,
                target_point_next=target_point_next)

        # ── student forward ───────────────────────────────────────────────────
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type='cuda'):
                student_out = student(
                    rgb, lidar_bev, target_point, ego_vel, command,
                    target_point_next=target_point_next, bbox_only=bbox_only)
                loss, comps = compute_kd_loss(
                    student, student_out, teacher_out, hooks, batch, device,
                    bbox_only=bbox_only, **loss_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer) 
            scaler.update()
        else:
            student_out = student(
                rgb, lidar_bev, target_point, ego_vel, command,
                target_point_next=target_point_next, bbox_only=bbox_only)
            loss, comps = compute_kd_loss(
                student, student_out, teacher_out, hooks, batch, device,
                bbox_only=bbox_only, **loss_weights)
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
    bbox_only: bool = False,
):
    student.eval()
    teacher.eval()

    running = {}
    n = 0

    for batch in tqdm(loader):
        rgb         = batch['rgb'].to(device)
        lidar_bev   = batch['lidar_bev'].to(device)

        if bbox_only:
            target_point, target_point_next, ego_vel, command = _make_dummy_planning_inputs(
                student.config, rgb.shape[0], device)
        else:
            target_point= batch['target_point'].to(device)
            target_point_next = None
            ego_vel     = batch['ego_vel'].to(device)
            command     = batch['command'].to(device)

        teacher_out = teacher(
            rgb, lidar_bev, target_point, ego_vel, command,
            target_point_next=target_point_next)

        # student in eval: kd dict is empty — run hooks-only path for val loss
        # temporarily switch to train so kd projectors fire
        student.train()
        student_out = student(
            rgb, lidar_bev, target_point, ego_vel, command,
            target_point_next=target_point_next, bbox_only=bbox_only)
        student.eval()

        _, comps = compute_kd_loss(
            student, student_out, teacher_out, hooks, batch, device,
            bbox_only=bbox_only, **loss_weights)

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
    bbox_only_train: bool = False,
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
    config.bbox_only_train = bool(bbox_only_train)

    if config.bbox_only_train and not config.detect_boxes:
        raise ValueError('bbox_only_train requires detect_boxes=True.')

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

    start_epoch = 0
    best_val_total = float('inf')

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    tb_dir = os.path.join(output_dir)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f'TensorBoard logs: {tb_dir}')

    # ── MODE DISPATCH: Fine-tune (continue_epoch=0) vs Resume (continue_epoch>0) ──
    mode = 'finetune' if int(continue_epoch) == 0 else 'resume'
    print(f'Checkpoint mode: {mode}')

    if mode == 'finetune':
        # ── FINE-TUNE MODE (continue_epoch=0) ──
        if loadfile is not None:
            if not os.path.isfile(loadfile):
                raise FileNotFoundError(loadfile)
            print(f'[FINETUNE] Loading model weights from: {loadfile}')
            
            # Load model weights only (no optimizer/scheduler/scaler/meta)
            student_state_dict = _load_finetuning_state(loadfile, device)
            
            # Handle bbox head initialization for fine-tune mode
            if config.detect_boxes:
                bbox_keys = _get_bbox_keys_from_state_dict(student_state_dict)
                if not bbox_keys:
                    # No bbox keys in loaded file: initialize from teacher
                    if not hasattr(teacher, 'head') or not hasattr(student, 'head'):
                        raise RuntimeError(
                            'detect_boxes=True but teacher/student bbox head unavailable. '
                            'Cannot fallback to teacher weights for missing bbox keys.'
                        )
                    print('[FINETUNE] Loaded checkpoint has no bbox head keys. '
                          'Initializing bbox head from teacher.')
                    # First init bbox from teacher
                    student.head.load_state_dict(teacher.head.state_dict(), strict=True)
                    # Then load checkpoint non-strictly to preserve bbox init but load other weights
                    student.load_state_dict(student_state_dict, strict=False)
                else:
                    # Bbox keys present: load strictly
                    student.load_state_dict(student_state_dict, strict=True)
            else:
                # No bbox detection: load strictly
                student.load_state_dict(student_state_dict, strict=True)
            
            print('[FINETUNE] Model loaded. Starting from epoch 0 (fine-tuning).')
        else:
            # No loadfile: random init + teacher bbox init (if enabled)
            if config.detect_boxes:
                if not hasattr(teacher, 'head') or not hasattr(student, 'head'):
                    raise RuntimeError('detect_boxes=True but teacher/student bbox head is unavailable.')
                student.head.load_state_dict(teacher.head.state_dict(), strict=True)
                print('Initialized student bbox head from teacher checkpoint weights.')
        
        # Fine-tune always resets to epoch 0 and fresh training state
        start_epoch = 0
        best_val_total = float('inf')
        
    else:
        # ── RESUME MODE (continue_epoch > 0) ──
        if loadfile is None:
            raise ValueError(
                'Resume mode (continue_epoch > 0) requires --loadfile argument. '
                'Checkpoint file must be provided.'
            )
        if not os.path.isfile(loadfile):
            raise FileNotFoundError(loadfile)
        
        print(f'[RESUME] Loading full checkpoint from: {loadfile}')
        
        # Load full state
        checkpoint = _load_resume_state(loadfile, device)
        
        # Validate detect_boxes compatibility from metadata
        saved_detect_boxes = checkpoint['hyperparameters'].get('detect_boxes', None)
        if saved_detect_boxes is None:
            raise ValueError(
                f'Resume mode requires hyperparameters.detect_boxes in checkpoint metadata. '
                f'File: {os.path.join(os.path.dirname(loadfile), "meta_data.json")}'
            )
        if bool(saved_detect_boxes) != bool(detect_boxes):
            raise ValueError(
                f'Resume mode: detect_boxes mismatch. '
                f'Checkpoint has detect_boxes={saved_detect_boxes}, '
                f'but runtime argument is detect_boxes={detect_boxes}. '
                f'These must match for resume to proceed.'
            )
        
        # Load student model
        student.load_state_dict(checkpoint['student_state_dict'], strict=True)
        
        # Load optimizer, scheduler, scaler
        if checkpoint.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Resolve resume epoch
        saved_epoch = checkpoint.get('epoch', None)
        inferred_epoch = _infer_epoch_from_path(loadfile)
        resume_epoch = saved_epoch if saved_epoch is not None else inferred_epoch
        if resume_epoch is None:
            resume_epoch = 0
        
        start_epoch = int(resume_epoch) + 1
        best_val_total = float(checkpoint.get('best_val_total', float('inf')))
        
        print(f'[RESUME] Epoch resolved from metadata: {saved_epoch}. '
              f'Continuing from epoch {start_epoch}.')


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
    hp.setdefault('bbox_only_train', bool(bbox_only_train))
    hp.setdefault('continue_epoch', int(continue_epoch))
    hp.setdefault('loss_weights', dict(lw))

    # ── training loop ─────────────────────────────────────────────────────────
    # Epochs are now zero-based: range [0, epochs)
    try:
        for epoch in range(start_epoch, epochs):
            print(f'\n[Epoch {epoch}/{epochs-1}]  lr={scheduler.get_last_lr()[0]:.2e}')

            train_metrics = train_one_epoch(
                teacher, student, train_loader, optimizer, device,
                hooks, lw, bbox_only=bbox_only_train,
                scaler=scaler, log_every=log_every)
            scheduler.step()

            val_metrics = validate(
                teacher, student, val_loader, device, hooks, lw,
                bbox_only=bbox_only_train)

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
