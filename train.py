"""CLI entrypoint for KD student training."""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import GlobalConfig
from .data import Custom_Data
from .kd_trainer import train as kd_train


DEFAULT_CONFIG_PATH = "/home/haemanth/Transfuser++/carla_garage/output/final_merged_model/config.json"
DEFAULT_TEACHER_CKPT = "/home/haemanth/Transfuser++/carla_garage/output/final_merged_model/model_final_merged.pth"
DEFAULT_OUTPUT_DIR = "/home/haemanth/Transfuser++/model_distillation/output"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the KD student model.")
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--teacher_ckpt", type=str, default=DEFAULT_TEACHER_CKPT)
    parser.add_argument("--data_root", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment_name", type=str, default="default_experiment")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--w_ck", type=float, default=1.0)
    parser.add_argument("--w_ts", type=float, default=1.0)
    parser.add_argument("--w_feat", type=float, default=0.5)
    parser.add_argument("--w_kd", type=float, default=0.5)
    parser.add_argument("--w_bbox", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--train_cache", type=str, default=None)
    parser.add_argument("--loadfile", type=str, default=None)
    parser.add_argument("--continue_epoch", type=int, default=0)
    parser.add_argument("--bbox_only_train", action="store_true")
    parser.add_argument("--detect_boxes", dest="detect_boxes", action="store_true")
    parser.add_argument("--no_detect_boxes", dest="detect_boxes", action="store_false")
    parser.set_defaults(detect_boxes=True)
    return parser.parse_args()


def build_config(config_path, detect_boxes, bbox_only_train=False):
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg_dict = json.load(handle)

    cfg_dict.pop("setting", None)
    config = GlobalConfig()
    config.initialize(setting="eval", **cfg_dict)

    config.compile = False
    config.use_semantic = False
    config.use_bev_semantic = False
    config.use_depth = False
    config.detect_boxes = bool(detect_boxes)
    config.use_wp_gru = False
    config.lidar_seq_len = 1
    config.augment = False
    config.use_ground_plane = False
    config.only_perception = False
    config.bbox_only_train = bool(bbox_only_train)
    return config


def np_route_to_checkpoints(route, predict_checkpoint_len):
    route = np.asarray(route, dtype=np.float32)
    if route.ndim != 2:
        route = route.reshape(-1, 2)
    if route.shape[0] >= predict_checkpoint_len:
        return route[:predict_checkpoint_len]
    pad_count = predict_checkpoint_len - route.shape[0]
    pad = np.repeat(route[-1:, :], pad_count, axis=0)
    return np.concatenate([route, pad], axis=0)


def make_collate_fn(predict_checkpoint_len, bbox_only=False):
    def collate_fn(batch_list):
        rgb = torch.stack([torch.from_numpy(item["rgb"]).float() for item in batch_list])
        lidar_bev = torch.stack([torch.from_numpy(item["lidar"]).float() for item in batch_list])
        if not bbox_only:
            target_point = torch.stack([torch.from_numpy(item["target_point"]).float() for item in batch_list])
            command = torch.stack([torch.from_numpy(item["command"]).float() for item in batch_list])
            ego_vel = torch.tensor([float(item["speed"]) for item in batch_list], dtype=torch.float32).unsqueeze(1)
            target_speed = torch.tensor([int(item["target_speed"]) for item in batch_list], dtype=torch.long)
            checkpoints = torch.stack([
                torch.from_numpy(np_route_to_checkpoints(item["route"], predict_checkpoint_len)).float()
                for item in batch_list
            ])
        center_heatmap_target = torch.stack([
            torch.from_numpy(item["center_heatmap_target"]).float() for item in batch_list
        ])
        wh_target = torch.stack([
            torch.from_numpy(item["wh_target"]).float() for item in batch_list
        ])
        yaw_class_target = torch.stack([
            torch.from_numpy(item["yaw_class_target"]).long() for item in batch_list
        ])
        yaw_res_target = torch.stack([
            torch.from_numpy(item["yaw_res_target"]).float() for item in batch_list
        ])
        offset_target = torch.stack([
            torch.from_numpy(item["offset_target"]).float() for item in batch_list
        ])
        velocity_target = torch.stack([
            torch.from_numpy(item["velocity_target"]).float() for item in batch_list
        ])
        brake_target = torch.stack([
            torch.from_numpy(item["brake_target"]).long() for item in batch_list
        ])
        pixel_weight = torch.stack([
            torch.from_numpy(item["pixel_weight"]).float() for item in batch_list
        ])
        avg_factor = torch.tensor([float(item["avg_factor"]) for item in batch_list], dtype=torch.float32)

        batch = {
            "rgb": rgb,
            "lidar_bev": lidar_bev,
            "center_heatmap_label": center_heatmap_target,
            "wh_label": wh_target,
            "yaw_class_label": yaw_class_target,
            "yaw_res_label": yaw_res_target,
            "offset_label": offset_target,
            "velocity_label": velocity_target,
            "brake_target_label": brake_target,
            "pixel_weight_label": pixel_weight,
            "avg_factor_label": avg_factor,
        }
        if not bbox_only:
            batch.update({
                "target_point": target_point,
                "ego_vel": ego_vel,
                "command": command,
                "checkpoints": checkpoints,
                "target_speed": target_speed,
            })
        return batch

    return collate_fn


def build_loaders(config, args):
    cache = {} if args.train_cache is not None else None
    train_dataset = Custom_Data(
        root=args.data_root,
        config=config,
        shared_dict=cache,
        validation=False,
    )
    val_dataset = Custom_Data(
        root=args.data_root,
        config=config,
        shared_dict=cache,
        validation=True,
    )

    if len(train_dataset) == 0:
        raise ValueError("No training samples were found. Check --data_root and the validation split naming.")
    if len(val_dataset) == 0:
        raise ValueError("No validation samples were found. Make sure some routes contain 'validation' in the name.")

    collate_fn = make_collate_fn(config.predict_checkpoint_len, bbox_only=args.bbox_only_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader


def main():
    args = parse_args()

    if not os.path.isfile(args.config_path):
        raise FileNotFoundError(args.config_path)
    if not os.path.isfile(args.teacher_ckpt):
        raise FileNotFoundError(args.teacher_ckpt)

    config = build_config(args.config_path, args.detect_boxes, args.bbox_only_train)
    train_loader, val_loader = build_loaders(config, args)

    run_output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(run_output_dir, exist_ok=True)
    print("KD training configuration:")
    print(f"  config_path : {args.config_path}")
    print(f"  teacher_ckpt: {args.teacher_ckpt}")
    print(f"  data_root   : {args.data_root}")
    print(f"  output_dir  : {run_output_dir}")
    print(f"  experiment  : {args.experiment_name}")
    print(f"  train/val   : {len(train_loader.dataset)} / {len(val_loader.dataset)} samples")
    print("  ego_vel     : forced to 0.0 by dataset design")
    print(f"  detect_boxes: {args.detect_boxes}")
    print(f"  bbox_only_train: {args.bbox_only_train}")
    
    # ── Mode semantics and guardrails ──
    if int(args.continue_epoch) == 0:
        print(f"\n  MODE: FINE-TUNING (continue_epoch=0)")
        if args.loadfile is not None:
            print(f"    ✓ loadfile provided: {args.loadfile}")
            print(f"    → Will load model weights only (no optimizer/scheduler/scaler)")
            print(f"    → Starting from epoch 0")
        else:
            print(f"    ✓ No loadfile: fresh training from random init")
            print(f"    → Starting from epoch 0")
    else:
        print(f"\n  MODE: RESUME (continue_epoch={args.continue_epoch})")
        if args.loadfile is None:
            raise ValueError(
                f"ERROR: Resume mode requires --loadfile argument. "
                f"Checkpoint file must be provided."
            )
        print(f"    ✓ loadfile required: {args.loadfile}")
        print(f"    → Will load full training state (model + optimizer + scheduler + scaler)")
        print(f"    → Will validate detect_boxes compatibility from metadata")
        print(f"    → Will continue from saved epoch")
    
    if args.loadfile is not None:
        print(f"  loadfile    : {args.loadfile}")
        print(f"  continue_epoch: {args.continue_epoch}")


    kd_train(
        hyperparameters={
            "config_path": args.config_path,
            "teacher_ckpt": args.teacher_ckpt,
            "data_root": list(args.data_root),
            "output_dir": run_output_dir,
            "experiment_name": args.experiment_name,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "num_workers": int(args.num_workers),
            "use_amp": bool(args.use_amp),
            "save_every": int(args.save_every),
            "log_every": int(args.log_every),
            "detect_boxes": bool(args.detect_boxes),
            "bbox_only_train": bool(args.bbox_only_train),
            "continue_epoch": int(args.continue_epoch),
            "loadfile": args.loadfile,
            "loss_weights": {
                "w_ck": float(args.w_ck),
                "w_ts": float(args.w_ts),
                "w_feat": float(args.w_feat),
                "w_kd": float(args.w_kd),
                "w_bbox": float(args.w_bbox),
                "T": float(args.temperature),
            },
        },
        config_path=args.config_path,
        teacher_ckpt=args.teacher_ckpt,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=run_output_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=args.use_amp,
        save_every=args.save_every,
        log_every=args.log_every,
        loss_weights={
            "w_ck": args.w_ck,
            "w_ts": args.w_ts,
            "w_feat": args.w_feat,
            "w_kd": args.w_kd,
            "w_bbox": args.w_bbox,
            "T": args.temperature,
        },
        detect_boxes=args.detect_boxes,
        bbox_only_train=args.bbox_only_train,
        loadfile=args.loadfile,
        continue_epoch=args.continue_epoch,
    )


if __name__ == "__main__":
    main()
