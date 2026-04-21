"""
Block-level inference timing for TransFuser++.

Run from the repo root:
    python -m model_nocarla.bench_timing

Outputs a sorted table showing average ms and % of total for every
instrumented block (backbone stages, GPT fusions, FPN, join, decoders, heads).
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from .model import LidarCenterNet
from .config import GlobalConfig
from . import transfuser_utils as t_u

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = "models/pretrained_models/all_towns"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WARMUP = 5    # frames discarded before timing starts
N_TIMED  = 20   # frames used to compute averages
BATCH_SIZE = 1

print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────
with open(os.path.join(CONFIG_PATH, "config.json"), "r") as f:
    cfg_dict = json.load(f)
cfg_dict.pop("setting", None)

config = GlobalConfig()
config.initialize(setting="eval", **cfg_dict)
config.compile = False
config.sync_batch_norm = False

net = None
for fname in os.listdir(CONFIG_PATH):
    if fname.startswith("model") and fname.endswith(".pth"):
        ckpt = os.path.join(CONFIG_PATH, fname)
        print(f"Loading: {ckpt}")
        net = LidarCenterNet(config)
        try:
            state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(state_dict, strict=True)
        net = net.to(DEVICE)
        net.eval()
        break

assert net is not None, "No checkpoint found in CONFIG_PATH"
print("Model loaded.\n")

# ──────────────────────────────────────────────────────────────────────────────
# Preprocess helpers
# ──────────────────────────────────────────────────────────────────────────────
def lidar_to_histogram_features(lidar_points):
    lidar = lidar_points[(lidar_points[:, 0] >= config.min_x) & (lidar_points[:, 0] < config.max_x)]
    lidar = lidar[(lidar[:, 1] >= config.min_y) & (lidar[:, 1] < config.max_y)]

    if config.use_ground_plane:
        below = lidar[lidar[:, 2] <= config.lidar_split_height]
        above = lidar[lidar[:, 2] > config.lidar_split_height]

        below_x = ((below[:, 0] - config.min_x) * config.pixels_per_meter).astype(np.int32)
        below_y = ((below[:, 1] - config.min_y) * config.pixels_per_meter).astype(np.int32)
        above_x = ((above[:, 0] - config.min_x) * config.pixels_per_meter).astype(np.int32)
        above_y = ((above[:, 1] - config.min_y) * config.pixels_per_meter).astype(np.int32)

        below_hist = np.zeros((config.lidar_resolution_height, config.lidar_resolution_width), dtype=np.float32)
        above_hist = np.zeros((config.lidar_resolution_height, config.lidar_resolution_width), dtype=np.float32)

        below_x = np.clip(below_x, 0, config.lidar_resolution_width - 1)
        below_y = np.clip(below_y, 0, config.lidar_resolution_height - 1)
        above_x = np.clip(above_x, 0, config.lidar_resolution_width - 1)
        above_y = np.clip(above_y, 0, config.lidar_resolution_height - 1)

        np.add.at(below_hist, (below_x, below_y), 1)
        np.add.at(above_hist, (above_x, above_y), 1)

        below_hist = np.clip(below_hist / config.hist_max_per_pixel, 0, 1).T
        above_hist = np.clip(above_hist / config.hist_max_per_pixel, 0, 1).T
        return np.stack([below_hist, above_hist], axis=0)

    pix_x = ((lidar[:, 0] - config.min_x) * config.pixels_per_meter).astype(np.int32)
    pix_y = ((lidar[:, 1] - config.min_y) * config.pixels_per_meter).astype(np.int32)
    hist = np.zeros((config.lidar_resolution_height, config.lidar_resolution_width), dtype=np.float32)
    pix_x = np.clip(pix_x, 0, config.lidar_resolution_width - 1)
    pix_y = np.clip(pix_y, 0, config.lidar_resolution_height - 1)
    np.add.at(hist, (pix_x, pix_y), 1)
    hist = np.clip(hist / config.hist_max_per_pixel, 0, 1).T
    return hist[np.newaxis, :, :]


def make_raw_inputs():
    image = np.zeros((config.camera_height, config.camera_width, 3), dtype=np.uint8)
    lidar_points = np.zeros((32768, 3), dtype=np.float32)
    target_point = np.zeros((2,), dtype=np.float32)
    velocity = 0.0
    command = 3
    return image, lidar_points, target_point, velocity, command


def preprocess_inputs(image, lidar_points, target_point, velocity, command, timer=None):
    if timer:
        with timer.measure("pre/image_to_tensor"):
            image = cv2.resize(image, (config.camera_width, config.camera_height))
            image = t_u.crop_array(config, image)
            rgb = torch.from_numpy(np.transpose(image.astype(np.float32), (2, 0, 1))).unsqueeze(0).to(DEVICE)
    else:
        image = cv2.resize(image, (config.camera_width, config.camera_height))
        image = t_u.crop_array(config, image)
        rgb = torch.from_numpy(np.transpose(image.astype(np.float32), (2, 0, 1))).unsqueeze(0).to(DEVICE)

    if timer:
        with timer.measure("pre/lidar_histogram_to_tensor"):
            lidar_feats = lidar_to_histogram_features(lidar_points)
            lidar_bev = torch.from_numpy(lidar_feats).unsqueeze(0).float().to(DEVICE)
    else:
        lidar_feats = lidar_to_histogram_features(lidar_points)
        lidar_bev = torch.from_numpy(lidar_feats).unsqueeze(0).float().to(DEVICE)

    if timer:
        with timer.measure("pre/target_point_tensor"):
            tp = torch.from_numpy(target_point).unsqueeze(0).to(DEVICE)
    else:
        tp = torch.from_numpy(target_point).unsqueeze(0).to(DEVICE)

    if timer:
        with timer.measure("pre/velocity_tensor"):
            vel = torch.tensor([[velocity]], dtype=torch.float32, device=DEVICE)
    else:
        vel = torch.tensor([[velocity]], dtype=torch.float32, device=DEVICE)

    if timer:
        with timer.measure("pre/command_tensor"):
            cmd = torch.from_numpy(t_u.command_to_one_hot(command)).unsqueeze(0).float().to(DEVICE)
    else:
        cmd = torch.from_numpy(t_u.command_to_one_hot(command)).unsqueeze(0).float().to(DEVICE)

    return rgb, lidar_bev, tp, vel, cmd


def postprocess_outputs(output, ego_velocity, timer=None):
    if timer:
        with timer.measure("post/extract_outputs"):
            pred_target_speed = output[1]
            pred_checkpoint = output[2]
            pred_bb_features = output[6]
    else:
        pred_target_speed = output[1]
        pred_checkpoint = output[2]
        pred_bb_features = output[6]

    if timer:
        with timer.measure("post/target_speed_decode"):
            probs = F.softmax(pred_target_speed, dim=1)[0].cpu().numpy() if pred_target_speed is not None else None
            idx = int(np.argmax(probs)) if probs is not None else 0
            target_speed = float(sum(p * s for p, s in zip(probs, config.target_speeds))) if (
                probs is not None and getattr(config, "use_twohot_target_speeds", False)
            ) else float(config.target_speeds[idx] if probs is not None else 0.0)
    else:
        probs = F.softmax(pred_target_speed, dim=1)[0].cpu().numpy() if pred_target_speed is not None else None
        idx = int(np.argmax(probs)) if probs is not None else 0
        target_speed = float(sum(p * s for p, s in zip(probs, config.target_speeds))) if (
            probs is not None and getattr(config, "use_twohot_target_speeds", False)
        ) else float(config.target_speeds[idx] if probs is not None else 0.0)

    if timer:
        with timer.measure("post/bbox_decode"):
            if getattr(config, "detect_boxes", False) and pred_bb_features is not None:
                _ = net.convert_features_to_bb_metric(pred_bb_features)
    else:
        if getattr(config, "detect_boxes", False) and pred_bb_features is not None:
            _ = net.convert_features_to_bb_metric(pred_bb_features)

    if timer:
        with timer.measure("post/control_pid"):
            pred_wp = pred_checkpoint
            if pred_wp is not None:
                vel_for_pid = torch.tensor([ego_velocity], dtype=torch.float32, device=pred_wp.device)
                _ = net.control_pid(pred_wp, vel_for_pid,
                                    tuned_aim_distance=False)
    else:
        pred_wp = pred_checkpoint
        if pred_wp is not None:
            vel_for_pid = torch.tensor([ego_velocity], dtype=torch.float32, device=pred_wp.device)
            _ = net.control_pid(pred_wp, vel_for_pid,
                                tuned_aim_distance=False)

    if timer:
        with timer.measure("post/checkpoint_to_numpy"):
            _ = pred_checkpoint[0].detach().cpu().numpy() if pred_checkpoint is not None else None
    else:
        _ = pred_checkpoint[0].detach().cpu().numpy() if pred_checkpoint is not None else None

# ──────────────────────────────────────────────────────────────────────────────
# Warmup  (no timer, just get CUDA kernels compiled/cached)
# ──────────────────────────────────────────────────────────────────────────────
print(f"Warming up ({N_WARMUP} iters) ...")
raw_image, raw_lidar_points, raw_tp, raw_vel, raw_cmd = make_raw_inputs()
with torch.inference_mode():
    for _ in range(N_WARMUP):
        rgb, lidar, tp, vel, cmd = preprocess_inputs(raw_image, raw_lidar_points, raw_tp, raw_vel, raw_cmd)
        output = net(rgb, lidar, tp, vel, cmd)
        postprocess_outputs(output, ego_velocity=raw_vel)
if torch.cuda.is_available():
    torch.cuda.synchronize()

# ──────────────────────────────────────────────────────────────────────────────
# Timed run with BlockTimer
# ──────────────────────────────────────────────────────────────────────────────
timer = t_u.BlockTimer()

print(f"Timing ({N_TIMED} iters) ...")
wall_times = []
pre_times = []
inf_times = []
post_times = []

with torch.inference_mode():
    for _ in range(N_TIMED):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_pre0 = time.perf_counter()
        rgb, lidar, tp, vel, cmd = preprocess_inputs(
            raw_image, raw_lidar_points, raw_tp, raw_vel, raw_cmd, timer=timer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pre_times.append((time.perf_counter() - t_pre0) * 1000.0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_inf0 = time.perf_counter()
        output = net(rgb, lidar, tp, vel, cmd, timer=timer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inf_times.append((time.perf_counter() - t_inf0) * 1000.0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_post0 = time.perf_counter()
        postprocess_outputs(output, ego_velocity=raw_vel, timer=timer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        post_times.append((time.perf_counter() - t_post0) * 1000.0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - t0) * 1000.0)

# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────
avg_wall = sum(wall_times) / len(wall_times)
print(f"\nWall-clock avg over {N_TIMED} iters: {avg_wall:.2f} ms  ({1000/avg_wall:.1f} FPS)")

avg_pre = sum(pre_times) / len(pre_times)
avg_inf = sum(inf_times) / len(inf_times)
avg_post = sum(post_times) / len(post_times)
print("\nStage totals (avg over timed iters):")
print(f"  Preprocess      : {avg_pre:.2f} ms")
print(f"  Model inference : {avg_inf:.2f} ms")
print(f"  Postprocess     : {avg_post:.2f} ms")

timer.report()
