import os
import json
import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

import numpy as np
from .model import LidarCenterNet
from .config import GlobalConfig
from pathlib import Path          # ✅ Add this
from PIL import Image             # ✅ Add this
import cv2
from . import transfuser_utils as t_u
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import cv2
from pathlib import Path
import time

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

        # 1️⃣ Create model on CPU
        net = LidarCenterNet(config)

        # 2️⃣ Load checkpoint to CPU (IMPORTANT)
        state_dict = torch.load(ckpt, map_location="cpu")

        # 3️⃣ Load weights while still on CPU
        net.load_state_dict(state_dict, strict=True)

        # 4️⃣ Move to GPU AFTER everything is loaded
        net = net.to(DEVICE)

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

import numpy as np
import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np


def plot_all_sensors_and_outputs(image, lidar_bev, results, config=None):
    """
    Plot all sensor data and model outputs in one figure.

    Args:
        image: RGB image [H,W,3] (numpy or torch)
        lidar_bev: BEV histogram [C,H,W] or [H,W] (numpy or torch)
        results: dict from postprocess_model_outputs
        config: optional
    """

    # ---------------------------
    # Utility: Safe tensor → numpy
    # ---------------------------
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    image = to_numpy(image)
    lidar_bev = to_numpy(lidar_bev)

    if image is not None:
        image = np.squeeze(image)

    if lidar_bev is not None:
        lidar_bev = np.squeeze(lidar_bev)

    plt.figure(figsize=(22, 12))

    # ---------------------------
    # 1. RGB Image
    # ---------------------------
    plt.subplot(2, 4, 1)
    if image is not None:
        plt.imshow(image)
    plt.title("Camera RGB")
    plt.axis("off")

    # ---------------------------
    # 2. LiDAR BEV
    # ---------------------------
    plt.subplot(2, 4, 2)
    if lidar_bev is not None:
        if lidar_bev.ndim == 3:
            if lidar_bev.shape[0] == 2:  # two-channel BEV
                plt.imshow(lidar_bev[0], cmap="Reds", origin="lower", alpha=0.6)
                plt.imshow(lidar_bev[1], cmap="Blues", origin="lower", alpha=0.6)
            else:
                plt.imshow(np.max(lidar_bev, axis=0), cmap="gray", origin="lower")
        else:
            plt.imshow(lidar_bev, cmap="gray", origin="lower")
    plt.title("LiDAR BEV Histogram")
    plt.axis("off")

    # ---------------------------
    # 3. Camera Semantic
    # ---------------------------
    plt.subplot(2, 4, 3)
    if results.get("pred_semantic") is not None:
        sem = to_numpy(results["pred_semantic"][0])
        sem_vis = np.argmax(sem, axis=0)
        plt.imshow(sem_vis)
    plt.title("Camera Semantic")
    plt.axis("off")

    # ---------------------------
    # 4. BEV Semantic
    # ---------------------------
    plt.subplot(2, 4, 4)
    if results.get("pred_bev_semantic") is not None:
        bev_sem = to_numpy(results["pred_bev_semantic"][0])
        bev_sem_vis = np.argmax(bev_sem, axis=0)
        plt.imshow(bev_sem_vis, origin="lower")
    plt.title("BEV Semantic")
    plt.axis("off")

    # ---------------------------
    # 5. Depth
    # ---------------------------
    plt.subplot(2, 4, 5)
    if results.get("pred_depth") is not None:
        depth = to_numpy(results["pred_depth"][0])
        plt.imshow(depth, cmap="plasma")
    plt.title("Predicted Depth")
    plt.axis("off")

    # ---------------------------
    # 6. Checkpoints / Waypoints
    # ---------------------------
    plt.subplot(2, 4, 6)
    if results.get("checkpoints") is not None:
        cps = to_numpy(results["checkpoints"])
        plt.plot(cps[:, 0], cps[:, 1], "o-", label="Checkpoints")
        plt.scatter(0, 0, c="red", label="Ego")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
    plt.title("Predicted Checkpoints / Waypoints")

    # ---------------------------
    # 7. Target Speed / Control Info
    # ---------------------------
    plt.subplot(2, 4, 7)
    ts = results.get("target_speed", 0)
    steer = results.get("steer", 0)
    throttle = results.get("throttle", 0)
    brake = results.get("brake", False)

    plt.axis("off")
    plt.text(0.1, 0.8, f"Target Speed: {ts:.2f} m/s", fontsize=12)
    plt.text(0.1, 0.6, f"Steer: {steer:.2f}", fontsize=12)
    plt.text(0.1, 0.4, f"Throttle: {throttle:.2f}", fontsize=12)
    plt.text(0.1, 0.2, f"Brake: {brake}", fontsize=12)
    plt.title("Control Outputs")

    # ---------------------------
    # 8. Bounding Boxes Info
    # ---------------------------
    plt.subplot(2, 4, 8)
    plt.axis("off")
    if results.get("pred_bb") is not None:
        plt.text(0.1, 0.5, "Bounding boxes exist", fontsize=12)
    else:
        plt.text(0.1, 0.5, "No bounding boxes", fontsize=12)
    plt.title("Bounding Boxes")

    plt.tight_layout()
    plt.show()
def plot_bev_with_bounding_boxes_centered(lidar_bev, results, config=None):
    """
    Plot LiDAR BEV with predicted bounding boxes and ego vehicle at the center of the BEV.
    
    Args:
        lidar_bev: BEV histogram [C,H,W] or [H,W] (numpy)
        results: dict returned by postprocess_model_outputs()
        config: optional, used for scaling if needed
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Determine BEV shape
    if lidar_bev.ndim == 3:
        H, W = lidar_bev.shape[1], lidar_bev.shape[2]
    else:
        H, W = lidar_bev.shape

    plt.figure(figsize=(8, 8))

    # Show LiDAR BEV
    if lidar_bev.ndim == 3:
        if lidar_bev.shape[0] == 2:
            plt.imshow(lidar_bev[0], cmap="Reds", origin="lower", alpha=0.6)
            plt.imshow(lidar_bev[1], cmap="Blues", origin="lower", alpha=0.4)
        else:
            plt.imshow(np.max(lidar_bev, axis=0), cmap="gray", origin="lower")
    else:
        plt.imshow(lidar_bev, cmap="gray", origin="lower")

    # Ego vehicle at center
    plt.scatter(W // 2, H // 2, c="green", s=50, label="Ego Vehicle")

    # Draw bounding boxes
    if results.get("pred_bb") is not None and config is not None:
        scale = config.pixels_per_meter
        half_W, half_H = W // 2, H // 2

        for bb in results["pred_bb"]:
            x, y, z, length, width, height, rx, ry, rz = bb
            # Convert world coordinates to BEV pixels
            bev_x = int(half_W + y * scale)  # BEV x = right
            bev_y = int(H - (half_H + x * scale))  # BEV y = forward

            rect = plt.Rectangle(
                (bev_x - int(length*scale/2), bev_y - int(width*scale/2)),
                int(length*scale),
                int(width*scale),
                linewidth=2,
                edgecolor="yellow",
                facecolor="none"
            )
            plt.gca().add_patch(rect)
            plt.text(bev_x, bev_y, "Obj", color="yellow", fontsize=8)

    plt.xlabel("Right (m)")
    plt.ylabel("Forward (m)")
    plt.title("LiDAR BEV with Bounding Boxes (Ego Centered)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

def postprocess_model_outputs(
    output,
    config,
    net,
    ego_velocity,
    compute_debug_output=False
):
    results = {}

    # --------------------------
    # Extract outputs
    # --------------------------
    pred_target_speed = output[1]       # [1, 8]
    pred_checkpoint = output[2]         # [1, 10, 2]
    pred_wp = pred_checkpoint
    pred_semantic = output[3]
    pred_bev_semantic = output[4]
    pred_depth = output[5]
    pred_bb_features = output[6]

    # --------------------------
    # Bounding boxes
    # --------------------------
    if getattr(config, "detect_boxes", False):
        pred_bounding_box = net.convert_features_to_bb_metric(pred_bb_features)
    else:
        pred_bounding_box = None

    results["pred_bb"] = pred_bounding_box

    # --------------------------
    # Target speed decoding
    # --------------------------
    if pred_target_speed is not None:
        probs = F.softmax(pred_target_speed, dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))

        if getattr(config, "use_twohot_target_speeds", False):
            target_speed = sum(
                p * s for p, s in zip(probs, config.target_speeds)
            )
        else:
            target_speed = config.target_speeds[idx]

        target_speed = float(target_speed)
    else:
        probs = None
        target_speed = 0.0

    results["target_speed"] = target_speed
    results["target_speed_probs"] = probs

    # --------------------------
    # Control (PID-direct or WP PID)
    # --------------------------
    if getattr(config, "inference_direct_controller", False) and pred_checkpoint is not None:

        # ✔ checkpoints MUST be NumPy
        checkpoints_np = pred_checkpoint[0].cpu().numpy()

        # ✔ velocity MUST be tensor [[v]]
        ego_velocity_tensor = torch.tensor(
            [[ego_velocity]],
            dtype=torch.float32,
            device=pred_checkpoint.device
        )

        # ✔ target speed tensor
        target_speed_tensor = torch.tensor(
            [[target_speed]],
            dtype=torch.float32,
            device=pred_checkpoint.device
        )

        steer, throttle, brake = net.control_pid_direct(
            checkpoints_np,
            target_speed_tensor,
            ego_velocity_tensor,
            ego_vehicle_location=np.array([0.0, 0.0], dtype=np.float32),
            ego_vehicle_rotation=0.0
        )

    else:
        steer, throttle, brake = net.control_pid(
            pred_wp,
            ego_velocity,
            tuned_aim_distance=False
        )

    # --------------------------
    # Safety stop
    # --------------------------
    if ego_velocity < 0.1:
        throttle = max(getattr(config, "creep_throttle", 0.1), throttle)
        brake = False

    results["steer"] = float(steer)
    results["throttle"] = float(throttle)
    results["brake"] = bool(brake)

    # --------------------------
    # Waypoints / checkpoints
    # --------------------------
    results["checkpoints"] = (
        pred_checkpoint[0].cpu().numpy() if pred_checkpoint is not None else None
    )
    results["waypoints"] = (
        pred_wp[0].cpu().numpy() if pred_wp is not None else None
    )

    # --------------------------
    # Additional outputs
    # --------------------------
    results["pred_semantic"] = pred_semantic
    results["pred_bev_semantic"] = pred_bev_semantic
    results["pred_depth"] = pred_depth

    return results


preprocessor = InputPreprocessor(config, device=DEVICE)


print('\n[2] Loading sensor data from files...')

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------
image_path = Path("carla_data_2/rgb/004905.jpg")
lidar_path = Path("carla_data_2/lidar/004905.npy")

# --------------------------------------------------
# LOAD RGB IMAGE
# --------------------------------------------------
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"    Image loaded: {image.shape}, dtype={image.dtype}")

# --------------------------------------------------
# LOAD LiDAR (.npy)
# --------------------------------------------------
lidar_raw = np.load(lidar_path)
print(f"    Raw LiDAR shape: {lidar_raw.shape}")

# Keep only XYZ
if lidar_raw.ndim != 2 or lidar_raw.shape[1] < 3:
    raise ValueError(f"Invalid LiDAR shape: {lidar_raw.shape}")

lidar_points = lidar_raw[:, :3].astype(np.float32)

print(f"    LiDAR XYZ shape: {lidar_points.shape}")

if not isinstance(lidar_points, np.ndarray):
    raise TypeError("Loaded LiDAR is not a numpy array")

if lidar_points.ndim != 2 or lidar_points.shape[1] != 3:
    raise ValueError(f"LiDAR must have shape (N,3), got {lidar_points.shape}")

lidar_points = lidar_points.astype(np.float32)
print(f"    LiDAR loaded: {lidar_points.shape}, dtype={lidar_points.dtype}")
command = 1   
target_point = [10.0, 1.0]   # meters in ego frame
velocity = 5.0

rgb, lidar_bev, tp, vel, cmd = preprocessor.prepare_inputs(
    image, lidar_points, target_point, velocity, command
)
total_inference_time = 0.0
total_frames = 0
# --------------------------------------------------
# FORWARD PASS
# --------------------------------------------------
with torch.inference_mode():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    output = net(rgb, lidar_bev, tp, vel, cmd)
    print("DEBUG: ",torch.cuda.current_device())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

inference_time = end_time - start_time
total_inference_time += inference_time
total_frames += 1

# Post-process into vehicle commands
ego_velocity = vel.item()  # current vehicle speed
control = postprocess_model_outputs(output, config, net, ego_velocity)


results = postprocess_model_outputs(output, config, net, ego_velocity=vel.item())

print("Steer:", results['steer'])
print("Throttle:", results['throttle'])
print("Brake:", results['brake'])
print("Target speed:", results['target_speed'])
print("Checkpoints shape:", results['checkpoints'].shape)
print("Waypoints shape:", results['waypoints'].shape)
print("Bounding Boxes:", results["pred_bb"])

#plot_all_sensors_and_outputs(image, lidar_bev.squeeze(), results, config=config)
#plot_bev_with_bounding_boxes_centered(lidar_bev.squeeze(), results, config=config)
if total_frames > 0:
    avg_inference_time = total_inference_time / total_frames
    model_fps = 1.0 / avg_inference_time

    print(f"Average inference time per frame: {avg_inference_time:.4f} seconds")
    print(f"Model inference FPS: {model_fps:.2f}")

