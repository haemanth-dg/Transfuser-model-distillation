# ==================================================
# Export LidarCenterNet to ONNX (ALL OUTPUTS)
# ==================================================

import os
import json
import torch
import torch.nn as nn
import onnx

from .model import LidarCenterNet
from .config import GlobalConfig

CONFIG_PATH = "models/pretrained_models/all_towns"
ONNX_PATH = "tfpp.onnx"
DEVICE = torch.device("cpu")

# --------------------------------------------------
# Load config
# --------------------------------------------------
with open(os.path.join(CONFIG_PATH, "config.json"), "r") as f:
    cfg_dict = json.load(f)

cfg_dict.pop("setting", None)

config = GlobalConfig()
config.initialize(setting="eval", **cfg_dict)
config.compile = False
config.sync_batch_norm = False

# --------------------------------------------------
# Load model
# --------------------------------------------------
net = None
for f in os.listdir(CONFIG_PATH):
    if f.startswith("model") and f.endswith(".pth"):
        net = LidarCenterNet(config).to(DEVICE)
        net.load_state_dict(
            torch.load(os.path.join(CONFIG_PATH, f), map_location=DEVICE),
            strict=True,
        )
        net.eval()
        break

assert net is not None
print("✅ Model loaded")

# --------------------------------------------------
# ONNX wrapper (FULL)
# --------------------------------------------------
class ONNXWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, rgb, lidar_bev, target_point, velocity, command):
        (
            pred_wp,
            pred_checkpoint,
            pred_target_speed_scalar,
            pred_target_speed,
            pred_semantic,
            pred_bev_semantic,
            pred_depth,
            pred_bb_features,
            attention_weights,
            selected_path,
        ) = self.net(rgb, lidar_bev, target_point, velocity, command)

        # -------------------------------------------------
        # HARD ANCHOR target_point (cannot be optimized out)
        # -------------------------------------------------
        tp_anchor = target_point.sum(dim=1, keepdim=True)  # (B, 1)

        pred_checkpoint = pred_checkpoint + 0.0 * tp_anchor.view(-1, 1, 1)
        pred_target_speed = pred_target_speed + 0.0 * tp_anchor
        pred_target_speed_scalar = pred_target_speed_scalar + 0.0 * tp_anchor[:, 0]

        # -------------------------------------------------
        # FINAL OUTPUTS
        # -------------------------------------------------
        return (
            pred_checkpoint,
            pred_target_speed_scalar,
            pred_target_speed,
            pred_semantic,
            pred_bev_semantic,
            pred_depth
        )

model = ONNXWrapper(net).eval()

# --------------------------------------------------
# Dummy inputs
# --------------------------------------------------
rgb = torch.randn(1, 3, 384, 1024)
lidar_bev = torch.randn(1, 1, 256, 256)
target_point = torch.randn(1, 2)
velocity = torch.randn(1, 1)
command = torch.zeros(1, 6)
command[0, 3] = 1.0

# --------------------------------------------------
# Export
# --------------------------------------------------
torch.onnx.export(
    model,
    (rgb, lidar_bev, target_point, velocity, command),
    ONNX_PATH,
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes=None,
    input_names=[
        "rgb",
        "lidar_bev",
        "target_point",
        "velocity",
        "command",
    ],
    output_names=[
        "pred_target_speed",
        "pred_checkpoint",
        "pred_semantic",
        "pred_bev_semantic",
        "pred_depth"    
        ],
)

onnx.checker.check_model(onnx.load(ONNX_PATH))
print("🎉 ONNX export with ALL outputs successful")
