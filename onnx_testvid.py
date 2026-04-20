import os
import json
import torch
import numpy as np
from .model import LidarCenterNet
from .config import GlobalConfig
from pathlib import Path          # ✅ Add this
from PIL import Image             # ✅ Add this
import cv2
from . import transfuser_utils as t_u
import torch.nn.functional as F
from copy import deepcopy
import time
import onnxruntime as ort

# --------------------------------------------------
# PATHS
# --------------------------------------------------
CONFIG_PATH = "models/pretrained_models/all_towns"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print("CUDA Device:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# LOAD CONFIG
# --------------------------------------------------
with open(os.path.join(CONFIG_PATH, "config.json"), "r") as f:
    cfg_dict = json.load(f)

# 🔑 REMOVE TRAINING-ONLY KEY
cfg_dict.pop("setting", None)

config = GlobalConfig()
config.initialize(setting="eval", **cfg_dict)

config.compile = False
config.sync_batch_norm = False

# --------------------------------------------------
# LOAD SINGLE MODEL
# --------------------------------------------------
net = None

for file in os.listdir(CONFIG_PATH):
    if file.startswith("model") and file.endswith(".pth"):
        ckpt = os.path.join(CONFIG_PATH, file)
        print("Loading:", ckpt)

        net = LidarCenterNet(config).to(DEVICE)
        state_dict = torch.load(ckpt, map_location=DEVICE)
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        break

assert net is not None, "No model checkpoint found"

print("✅ Model loaded successfully on GPU")

def lidar_to_histogram_features(lidar, config):
    """
    Convert LiDAR point cloud to BEV histogram features.
    
    Args:
        lidar: numpy array of shape (N, 3) with x, y, z coordinates in ego frame
               x = forward, y = right, z = up
        config: GlobalConfig object
    
    Returns:
        BEV histogram tensor of shape (C, H, W) where:
        - C = 1 if use_ground_plane is False (only above ground)
        - C = 2 if use_ground_plane is True (below + above ground)
    """
    # Filter by range first
    lidar = lidar[(lidar[:, 0] >= config.min_x) & (lidar[:, 0] < config.max_x)]
    lidar = lidar[(lidar[:, 1] >= config.min_y) & (lidar[:, 1] < config.max_y)]
    
    # Create histogram for all points (single channel if no ground plane separation)
    if config.use_ground_plane:
        # Two channel mode: separate above and below ground
        below = lidar[lidar[:, 2] <= config.lidar_split_height]
        above = lidar[lidar[:, 2] > config.lidar_split_height]
        
        # Convert to pixel coordinates
        below_pixels_x = ((below[:, 0] - config.min_x) * config.pixels_per_meter).astype(np.int32)
        below_pixels_y = ((below[:, 1] - config.min_y) * config.pixels_per_meter).astype(np.int32)
        above_pixels_x = ((above[:, 0] - config.min_x) * config.pixels_per_meter).astype(np.int32)
        above_pixels_y = ((above[:, 1] - config.min_y) * config.pixels_per_meter).astype(np.int32)
        
        # Create histograms
        below_histogram = np.zeros((config.lidar_resolution_height, config.lidar_resolution_width), dtype=np.float32)
        above_histogram = np.zeros((config.lidar_resolution_height, config.lidar_resolution_width), dtype=np.float32)
        
        # Clip to valid range
        below_pixels_x = np.clip(below_pixels_x, 0, config.lidar_resolution_width - 1)
        below_pixels_y = np.clip(below_pixels_y, 0, config.lidar_resolution_height - 1)
        above_pixels_x = np.clip(above_pixels_x, 0, config.lidar_resolution_width - 1)
        above_pixels_y = np.clip(above_pixels_y, 0, config.lidar_resolution_height - 1)
        
        # Count points per pixel
        np.add.at(below_histogram, (below_pixels_x, below_pixels_y), 1)
        np.add.at(above_histogram, (above_pixels_x, above_pixels_y), 1)
        
        # Normalize by max hits per pixel
        below_histogram = np.clip(below_histogram / config.hist_max_per_pixel, 0, 1)
        above_histogram = np.clip(above_histogram / config.hist_max_per_pixel, 0, 1)
        
        # Transpose to get correct orientation (x forward, y right)
        below_histogram = below_histogram.T
        above_histogram = above_histogram.T
        
        # Stack channels: [below, above]
        features = np.stack([below_histogram, above_histogram], axis=0)
    else:
        # Single channel mode: all points combined
        pixels_x = ((lidar[:, 0] - config.min_x) * config.pixels_per_meter).astype(np.int32)
        pixels_y = ((lidar[:, 1] - config.min_y) * config.pixels_per_meter).astype(np.int32)
        
        # Create histogram
        histogram = np.zeros((config.lidar_resolution_height, config.lidar_resolution_width), dtype=np.float32)
        
        # Clip to valid range
        pixels_x = np.clip(pixels_x, 0, config.lidar_resolution_width - 1)
        pixels_y = np.clip(pixels_y, 0, config.lidar_resolution_height - 1)
        
        # Count points per pixel
        np.add.at(histogram, (pixels_x, pixels_y), 1)
        
        # Normalize by max hits per pixel
        histogram = np.clip(histogram / config.hist_max_per_pixel, 0, 1)
        
        # Transpose to get correct orientation (x forward, y right)
        histogram = histogram.T
        
        # Add channel dimension: shape becomes (1, H, W)
        features = histogram[np.newaxis, :, :]
    
    return features

class InputPreprocessor:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device

    def preprocess_image(self, image):
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        image = cv2.resize(image, (self.config.camera_width, self.config.camera_height))
        image = t_u.crop_array(self.config, image)
        image = torch.from_numpy(np.transpose(image.astype(np.float32), (2, 0, 1))).unsqueeze(0).to(self.device)
        return image

    def preprocess_lidar(self, lidar_points):
        features = lidar_to_histogram_features(lidar_points, self.config)
        if features.ndim == 2:
            features = features[None, :, :]
        features = torch.from_numpy(features).unsqueeze(0).float().to(self.device)
        return features

    def preprocess_target_point(self, target_point):
        tp = torch.from_numpy(np.array(target_point, dtype=np.float32)).unsqueeze(0).to(self.device)
        return tp

    def preprocess_velocity(self, velocity):
        return torch.tensor([[velocity]], dtype=torch.float32).to(self.device)

    def preprocess_command(self, command):
        cmd = torch.from_numpy(t_u.command_to_one_hot(command)).unsqueeze(0).float().to(self.device)
        return cmd

    # ----------------------------------------
    # Prepare all inputs in one call
    # ----------------------------------------
    def prepare_inputs(self, image, lidar_points, target_point, velocity, command):
        """
        Returns a tuple of tensors ready for LidarCenterNet.forward():
        (rgb, lidar_bev, target_point, velocity, command)
        Shapes:
            rgb: [1, 3, H, W]
            lidar_bev: [1, 1, 256, 256]
            target_point: [1, 2]
            velocity: [1, 1]
            command: [1, 6]
        """
        rgb = self.preprocess_image(image)
        lidar_bev = self.preprocess_lidar(lidar_points)
        tp = self.preprocess_target_point(target_point)
        vel = self.preprocess_velocity(velocity)
        cmd = self.preprocess_command(command)
        return rgb, lidar_bev, tp, vel, cmd

def postprocess_model_outputs(
    output,
    config,
    net,
    ego_velocity,
    compute_debug_output=False
):
    results = {}

    # -------------------------------------------------
    # Helper: convert everything to NumPy (CPU only)
    # -------------------------------------------------
    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    # -------------------------------------------------
    # Extract outputs (force CPU numpy)
    # -------------------------------------------------
    pred_target_speed = to_numpy(output[1])  # [1, 8]
    pred_checkpoint   = to_numpy(output[2])  # [1, 10, 2]
    pred_semantic     = to_numpy(output[3])
    pred_bev_semantic = to_numpy(output[4])
    pred_depth        = to_numpy(output[5])

    pred_wp = pred_checkpoint

    # -------------------------------------------------
    # Target speed decoding (ONNX shape [1,1,8])
    # -------------------------------------------------
    if pred_target_speed is not None:

        # shape: (1,1,8) → (8,)
        logits = pred_target_speed[0, 0]

        # stable softmax over last dim
        logits = logits - np.max(logits)
        exp_x = np.exp(logits)
        probs = exp_x / np.sum(exp_x)

        idx = int(np.argmax(probs))

        if getattr(config, "use_twohot_target_speeds", False):
            target_speed = float(
                np.dot(probs, np.array(config.target_speeds, dtype=np.float32))
            )
        else:
            target_speed = float(config.target_speeds[idx])

    else:
        probs = None
        target_speed = 0.0


    results["target_speed"] = target_speed
    results["target_speed_probs"] = probs

    # -------------------------------------------------
    # Control (CPU only)
    # -------------------------------------------------
    if getattr(config, "inference_direct_controller", False) and pred_checkpoint is not None:

        checkpoints_np = pred_checkpoint[0]  # already NumPy

        # Everything stays NumPy
        steer, throttle, brake = net.control_pid_direct(
            checkpoints_np,
            np.array([[target_speed]], dtype=np.float32),
            np.array([[ego_velocity]], dtype=np.float32),
            ego_vehicle_location=np.array([0.0, 0.0], dtype=np.float32),
            ego_vehicle_rotation=0.0
        )

    else:
        # If control_pid expects torch, convert ONCE here
        if isinstance(pred_wp, np.ndarray):
            pred_wp_tensor = torch.from_numpy(pred_wp)
        else:
            pred_wp_tensor = pred_wp

        steer, throttle, brake = net.control_pid(
            pred_wp_tensor,
            float(ego_velocity),
            tuned_aim_distance=False
        )

    # -------------------------------------------------
    # Safety stop
    # -------------------------------------------------
    if ego_velocity < 0.1:
        throttle = max(getattr(config, "creep_throttle", 0.1), throttle)
        brake = False

    results["steer"] = float(steer)
    results["throttle"] = float(throttle)
    results["brake"] = bool(brake)

    # -------------------------------------------------
    # Waypoints / checkpoints (already CPU)
    # -------------------------------------------------
    results["checkpoints"] = (
        pred_checkpoint[0] if pred_checkpoint is not None else None
    )

    results["waypoints"] = (
        pred_wp[0] if pred_wp is not None else None
    )

    # -------------------------------------------------
    # Additional outputs (already NumPy)
    # -------------------------------------------------
    results["pred_semantic"] = pred_semantic
    results["pred_bev_semantic"] = pred_bev_semantic
    results["pred_depth"] = pred_depth

    return results


import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
preprocessor = InputPreprocessor(config, device=DEVICE)
# -----------------------------
# USER CONFIGURATION
# -----------------------------
LEFT_ROOT = "idd/primary/d1"
LEFT_IMG_DIR = os.path.join(LEFT_ROOT, "leftCamImgs")
LIDAR_DIR = "idd/lidar/d1"
CSV_PATH = os.path.join(LEFT_ROOT, "train.csv")

IMAGE_EXT = ".jpg"
PAD = 7

TARGET_POINT = [10.0, 1.0]
VELOCITY = 5.0
COMMAND = 0

VIDEO_PATH = "d1_sequence_fast.mp4"
FPS = 10

# preprocessor, net, config, postprocess_model_outputs must be defined
# -----------------------------
# LOAD CSV
# -----------------------------
df = pd.read_csv(CSV_PATH)
image_indices = df["image_idx"].values

# -----------------------------
# Initialize VideoWriter
# -----------------------------
video_writer = None
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def render_frame_fast(image_left, image_right, lidar_bev, results, config=None):
    """
    Render sensor outputs to a numpy RGB array using matplotlib canvas.
    Layout:
    Row 1: Camera L | Camera R (+controls overlay) | LiDAR BEV (rotated CCW)
    Row 2: Depth | Camera Semantic | Predicted Waypoints

    Returns:
        frame: np.uint8 RGB image (H,W,3)
    """

    # -------------------------------------------------
    # Utility: Safe tensor → numpy (handles CUDA)
    # -------------------------------------------------
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    image_left = to_numpy(image_left)
    image_right = to_numpy(image_right)
    lidar_bev = to_numpy(lidar_bev)

    if image_left is not None:
        image_left = np.squeeze(image_left)

    if image_right is not None:
        image_right = np.squeeze(image_right)

    if lidar_bev is not None:
        lidar_bev = np.squeeze(lidar_bev)

    # -------------------------------------------------
    # Create figure
    # -------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # -------------------------
    # Camera Left
    # -------------------------
    if image_left is not None:
        axes[0, 0].imshow(image_left)
    axes[0, 0].set_title("Camera Input")
    axes[0, 0].axis("off")

    # -------------------------
    # Camera Right
    # -------------------------
    if image_right is not None:
        axes[0, 1].imshow(image_right)
    axes[0, 1].set_title("Model Control")
    axes[0, 1].axis("off")

    # Overlay control info
    ts = results.get("target_speed", 0.0)
    steer = results.get("steer", 0.0)
    throttle = results.get("throttle", 0.0)
    brake = results.get("brake", False)

    overlay_text = (
        f"Target Speed: {ts:.2f} m/s\n"
        f"Steer: {steer:.2f}\n"
        f"Throttle: {throttle:.2f}\n"
        f"Brake: {brake}"
    )

    box_color = "red" if brake else "black"
    box_alpha = 0.6 if brake else 0.5

    axes[0, 1].text(
        0.02, 0.98,
        overlay_text,
        transform=axes[0, 1].transAxes,
        fontsize=12,
        verticalalignment="top",
        color="white",
        bbox=dict(
            facecolor=box_color,
            alpha=box_alpha,
            edgecolor="white",
            linewidth=2
        )
    )

    # -------------------------
    # LiDAR BEV
    # -------------------------
    if lidar_bev is not None:
        if lidar_bev.ndim == 3:
            lidar_img = np.max(lidar_bev, axis=0)
        else:
            lidar_img = lidar_bev

        lidar_img = np.rot90(lidar_img, k=3)
        axes[0, 2].imshow(lidar_img, cmap="gray", origin="lower")

    axes[0, 2].set_title("LiDAR BEV")
    axes[0, 2].axis("off")

    # -------------------------
    # Depth
    # -------------------------
    depth = results.get("pred_depth")
    if depth is not None:
        depth = to_numpy(depth[0])
        depth = np.squeeze(depth)

        axes[1, 0].imshow(depth, cmap="plasma")
        axes[1, 0].set_title("Predicted Depth")
        axes[1, 0].axis("off")
    else:
        axes[1, 0].axis("off")

    # -------------------------
    # Camera Semantic
    # -------------------------
    sem = results.get("pred_semantic")
    if sem is not None:
        sem = to_numpy(sem[0])
        sem = np.squeeze(sem)

        axes[1, 1].imshow(np.argmax(sem, axis=0))
        axes[1, 1].set_title("Camera Semantic")
        axes[1, 1].axis("off")
    else:
        axes[1, 1].axis("off")

    # -------------------------
    # Waypoints
    # -------------------------
    cps = results.get("checkpoints")
    if cps is not None:
        cps = to_numpy(cps)

        cps_rot = np.zeros_like(cps)
        cps_rot[:, 0] = -cps[:, 1]
        cps_rot[:, 1] =  cps[:, 0]

        axes[1, 2].plot(
            cps_rot[:, 0],
            cps_rot[:, 1],
            "o-",
            label="Waypoints"
        )
        axes[1, 2].scatter(0, 0, c="red", label="Ego")
        axes[1, 2].axis("equal")
        axes[1, 2].grid(True)
        axes[1, 2].legend()
    else:
        axes[1, 2].axis("off")

    # -------------------------------------------------
    # Convert matplotlib figure → numpy RGB frame
    # -------------------------------------------------
    plt.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()

    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    frame = buf[:, :, :3].copy()  # drop alpha

    plt.close(fig)

    return frame



total_inference_time = 0.0
total_frames = 0
max_frames = 50
processed_frames = 0
ONNX_PATH = "tfpp.onnx"
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True
sess = ort.InferenceSession(ONNX_PATH, sess_options=sess_options, providers=["CUDAExecutionProvider"])

for image_idx in tqdm(image_indices, desc="Generating video"):
    if processed_frames >= max_frames:
        break

    idx = str(image_idx).zfill(PAD)
    image_path = os.path.join(LEFT_IMG_DIR, idx + IMAGE_EXT)
    lidar_path = os.path.join(LIDAR_DIR, idx + ".npy")

    if not os.path.exists(image_path) or not os.path.exists(lidar_path):
        continue

    # Load sensors
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    lidar_points = np.load(lidar_path)[:, :3].astype(np.float32)

    # Preprocess inputs
    rgb, lidar_bev, tp, vel, cmd = preprocessor.prepare_inputs(
        image, lidar_points, TARGET_POINT, VELOCITY, COMMAND
    )

    # Convert PyTorch tensors to numpy for ONNX Runtime
    ort_inputs = {
        "rgb": rgb.cpu().numpy(),
        "lidar_bev": lidar_bev.cpu().numpy(),
        "target_point": tp.cpu().numpy(),
        "velocity": vel.cpu().numpy(),
        "command": cmd.cpu().numpy(),
    }

    # ONNX inference
    start_time = time.time()

    ort_outputs = sess.run(None, ort_inputs)
    end_time = time.time()

    inference_time = end_time - start_time
    total_inference_time += inference_time
    total_frames += 1
    processed_frames += 1

    # Map ONNX outputs to expected format
    output = (
        None,                                   # unused
        ort_outputs[0],                          # pred_target_speed
        ort_outputs[1],                          # pred_checkpoint
        ort_outputs[2],                          # pred_semantic
        ort_outputs[3],                          # pred_bev_semantic
        ort_outputs[4],                          # pred_depth
        None,                                   # unused
    )

    # Postprocess vehicle commands
    results = postprocess_model_outputs(output, config, net, ego_velocity=vel.item())

    # Render frame
    frame_rgb = render_frame_fast(image, image, lidar_bev.squeeze(), results, config=config)

    # Initialize video writer once
    if video_writer is None:
        h, w = frame_rgb.shape[:2]
        video_writer = cv2.VideoWriter(
            VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h)
        )

    # Write frame (convert RGB -> BGR)
    video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

# Finish video
if video_writer is not None:
    video_writer.release()
    print(f"Fast video saved at: {VIDEO_PATH}")
if total_frames > 0:
    avg_inference_time = total_inference_time / total_frames
    model_fps = 1.0 / avg_inference_time

    print(f"Average inference time per frame: {avg_inference_time:.4f} seconds")
    print(f"Model inference FPS: {model_fps:.2f}")
