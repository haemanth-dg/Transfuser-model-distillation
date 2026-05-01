# ==================================================
# Export StudentNet to ONNX (inference outputs)
# ==================================================

import json
import os

import onnx
import torch
import torch.nn as nn

from .config import GlobalConfig
from .student_model import StudentNet

CONFIG_PATH = "student_model"
ONNX_PATH = "student_finetuned.onnx"
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
# Student ONNX export uses forward_inference outputs only.
# Disable bbox head to avoid tracing detection branch.
config.detect_boxes = False


# --------------------------------------------------
# Load model
# --------------------------------------------------
ckpt_path = os.path.join(CONFIG_PATH, "student_best.pth")
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

net = StudentNet(config, use_kd_projectors=False).to(DEVICE)

try:
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
except TypeError:
    state_dict = torch.load(ckpt_path, map_location=DEVICE)

missing, unexpected = net.load_state_dict(state_dict, strict=False)
if missing:
    print(f"Warning: missing keys while loading student checkpoint: {len(missing)}")
if unexpected:
    print(f"Warning: unexpected keys while loading student checkpoint: {len(unexpected)}")

net.eval()
print("Model loaded")

# Replace adaptive pooling in fusion blocks with fixed-size resize to avoid
# legacy ONNX exporter failures on adaptive_avg_pool2d shape inference.
for fusion in net.backbone.fusions:
    fusion.img_pool = nn.Upsample(size=fusion.img_anchor, mode="bilinear", align_corners=False)
    fusion.lidar_pool = nn.Upsample(size=fusion.lidar_anchor, mode="bilinear", align_corners=False)


# --------------------------------------------------
# ONNX wrapper
# --------------------------------------------------
class ONNXWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, rgb, lidar_bev, target_point, velocity, command):
        pred_target_speed, pred_checkpoint = self.net.forward_inference(
            rgb, lidar_bev, target_point, velocity, command
        )
        return pred_target_speed, pred_checkpoint


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
    ],
)

onnx.checker.check_model(onnx.load(ONNX_PATH))
print("ONNX export for student model successful")