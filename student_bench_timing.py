"""
Block-level inference timing for StudentNet.

Run from the repo root:
    python -m model_nocarla.student_bench_timing

Outputs a sorted table showing average ms and % of total for every
instrumented block (backbone stages, join, decoders, heads).
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .student_model import StudentNet
from .config import GlobalConfig
from . import transfuser_utils as t_u

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = "student_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WARMUP = 5    # frames discarded before timing starts
N_TIMED  = 20   # frames used to compute averages
BATCH_SIZE = 1

print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────────────────────────────────────
# Load config and model
# ──────────────────────────────────────────────────────────────────────────────
with open(os.path.join(CONFIG_PATH, "config.json"), "r") as f:
    cfg_dict = json.load(f)
cfg_dict.pop("setting", None)

config = GlobalConfig()
config.initialize(setting="eval", **cfg_dict)
config.compile = False
config.sync_batch_norm = False
# Disable bbox for benchmarking to avoid shape mismatch issues
config.detect_boxes = False

net = None
ckpt_path = os.path.join(CONFIG_PATH, "student_best.pth")
if os.path.exists(ckpt_path):
    print(f"Loading: {ckpt_path}")
    net = StudentNet(config, use_kd_projectors=False)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    # Filter out bbox head weights since we disabled detect_boxes
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
    
    net.load_state_dict(state_dict, strict=False)
    net = net.to(DEVICE)
    net.eval()

assert net is not None, f"No checkpoint found at {ckpt_path}"
print("Model loaded.\n")

# ──────────────────────────────────────────────────────────────────────────────
# Dummy inputs (matches real shapes used in test.py)
# ──────────────────────────────────────────────────────────────────────────────
def make_inputs():
    C_lidar = 2 if config.use_ground_plane else config.lidar_seq_len
    img_h = getattr(config, "cropped_height", config.camera_height)
    img_w = getattr(config, "cropped_width",  config.camera_width)
    rgb          = torch.zeros(BATCH_SIZE, 3, img_h, img_w, device=DEVICE)
    lidar_bev    = torch.zeros(BATCH_SIZE, C_lidar,
                               config.lidar_resolution_height, config.lidar_resolution_width,
                               device=DEVICE)
    target_point = torch.zeros(BATCH_SIZE, 2, device=DEVICE)
    ego_vel      = torch.zeros(BATCH_SIZE, 1, device=DEVICE)
    command      = torch.zeros(BATCH_SIZE, 6, device=DEVICE)
    command[:, 1] = 1.0   # straight command
    return rgb, lidar_bev, target_point, ego_vel, command

rgb, lidar_bev, target_point, ego_vel, command = make_inputs()

# ──────────────────────────────────────────────────────────────────────────────
# Warmup  (no timer, just get CUDA kernels compiled/cached)
# ──────────────────────────────────────────────────────────────────────────────
print(f"Warming up ({N_WARMUP} iters) ...")
with torch.inference_mode():
    for _ in range(N_WARMUP):
        _ = net(rgb, lidar_bev, target_point, ego_vel, command)
if torch.cuda.is_available():
    torch.cuda.synchronize()

# ──────────────────────────────────────────────────────────────────────────────
# Timed run with BlockTimer
# ──────────────────────────────────────────────────────────────────────────────
timer = t_u.BlockTimer()

print(f"Timing ({N_TIMED} iters) ...")
wall_times = []

with torch.inference_mode():
    for _ in range(N_TIMED):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _ = net(rgb, lidar_bev, target_point, ego_vel, command, timer=timer)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - t0) * 1000.0)

# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────
avg_wall = sum(wall_times) / len(wall_times)
print(f"\nWall-clock avg over {N_TIMED} iters: {avg_wall:.2f} ms  ({1000/avg_wall:.1f} FPS)")

timer.report()
