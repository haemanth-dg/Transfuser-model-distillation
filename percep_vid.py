import os
import json
import gzip
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
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

CONFIG_PATH = "models/pretrained_models/all_towns"


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = resolve_device()

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
    if file.startswith("model_final") and file.endswith(".pth"):
        ckpt = os.path.join(CONFIG_PATH, file)
        print("Loading:", ckpt)

        net = LidarCenterNet(config).to(DEVICE)
        state_dict = torch.load(ckpt, map_location=DEVICE)
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        break

assert net is not None, "No model checkpoint found"

print(f"✅ Model loaded successfully on {DEVICE}")

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

# def lidar_to_histogram_features(lidar, config):
#     """
#     Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
#     :param lidar: (N,3) numpy, LiDAR point cloud
#     :param use_ground_plane, whether to use the ground plane
#     :return: (2, H, W) numpy, LiDAR as sparse image
#     """
#     use_ground_plane = config.use_ground_plane
#     def splat_points(point_cloud):
#       # 256 x 256 grid
#       xbins = np.linspace(config.min_x, config.max_x,
#                           (config.max_x - config.min_x) * int(config.pixels_per_meter) + 1)
#       ybins = np.linspace(config.min_y, config.max_y,
#                           (config.max_y - config.min_y) * int(config.pixels_per_meter) + 1)
#       hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
#       hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
#       overhead_splat = hist / config.hist_max_per_pixel
#       # The transpose here is an efficient axis swap.
#       # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
#       # (x height channel, y width channel)
#       return overhead_splat.T

#     # Remove points above the vehicle
#     lidar = lidar[lidar[..., 2] < config.max_height_lidar]
#     below = lidar[lidar[..., 2] <= config.lidar_split_height]
#     above = lidar[lidar[..., 2] > config.lidar_split_height]
#     below_features = splat_points(below)
#     above_features = splat_points(above)
#     if use_ground_plane:
#       features = np.stack([below_features, above_features], axis=-1)
#     else:
#       features = np.stack([above_features], axis=-1)
#     features = np.transpose(features, (2, 0, 1)).astype(np.float32)
#     return features


class InputPreprocessor:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device if device is not None else DEVICE

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
    ego_velocity_tensor=None,
    compute_debug_output=False
):
    results = {}

    if torch.is_tensor(ego_velocity):
        ego_velocity_scalar = float(ego_velocity.detach().flatten()[0].item())
    else:
        ego_velocity_scalar = float(ego_velocity)

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
        if ego_velocity_tensor is None:
            ego_velocity_tensor = torch.tensor(
                [[ego_velocity_scalar]],
                dtype=torch.float32,
                device=pred_checkpoint.device
            )
        else:
            ego_velocity_tensor = ego_velocity_tensor.to(pred_checkpoint.device)

        steer, throttle, brake = net.control_pid_direct(
            checkpoints_np,
            float(target_speed),
            ego_velocity_tensor,
            ego_vehicle_location=np.array([0.0, 0.0], dtype=np.float32),
            ego_vehicle_rotation=0.0
        )

    else:
        if ego_velocity_tensor is None:
            ego_velocity_for_pid = torch.tensor(
                [[ego_velocity_scalar]],
                dtype=torch.float32,
                device=pred_wp.device
            )
        else:
            ego_velocity_for_pid = ego_velocity_tensor.to(pred_wp.device)

        steer, throttle, brake = net.control_pid(
            pred_wp,
            ego_velocity_for_pid,
            tuned_aim_distance=False
        )

    # --------------------------
    # Safety stop
    # --------------------------
    if ego_velocity_scalar < 0.1:
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
# -----------------------------
# USER CONFIGURATION
# -----------------------------

IMAGE_EXT = ".jpg"
PAD = 7

VIDEO_PATH = "videos/idd.mp4"
FPS = 10

# -----------------------------
# Initialize VideoWriter
# -----------------------------
video_writer = None
def render_frame_fast(image_left, image_right, lidar_bev, results, route, target_point_xy, command=4, config=None):
    """
    Render sensor outputs to a numpy RGB array using matplotlib canvas.
    Layout:
    Row 1: Camera L | Camera R (+controls overlay) | LiDAR BEV (rotated CCW)
    Row 2: Depth (inverted) | Camera Semantic | Predicted Waypoints
    """

    if torch.is_tensor(lidar_bev):
        lidar_bev = lidar_bev.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # -------------------------
    # --- Camera Left (RGB) ---
    # -------------------------
    axes[0, 0].imshow(image_left)
    axes[0, 0].set_title("Camera Input")
    axes[0, 0].axis("off")

    # --------------------------
    # --- Camera Right (RGB) ---
    # --------------------------
    axes[0, 1].imshow(image_right)
    axes[0, 1].set_title("Model Control")
    axes[0, 1].axis("off")

    # --- Overlay control info on second camera ---
    ts = results.get("target_speed", 0.0)
    steer = results.get("steer", 0.0)
    throttle = results.get("throttle", 0.0)
    brake = results.get("brake", False)

    command_names = {
        1: "LEFT",
        2: "RIGHT",
        4: "LANEFOLLOW",
    }
    command_str = command_names.get(command, "UNKNOWN")

    target_point_xy = np.asarray(target_point_xy, dtype=np.float32).reshape(-1)
    if target_point_xy.shape[0] < 2:
        target_point_xy = np.array([0.0, 0.0], dtype=np.float32)
    target_point_forward = float(target_point_xy[0])
    target_point_right = float(target_point_xy[1])
    target_point_distance = float(np.linalg.norm(target_point_xy[:2]))

    overlay_text = (
        f"Command: {command_str}\n"
        f"Target point (fwd/right): [{target_point_forward:.2f}, {target_point_right:.2f}] m\n"
        f"Target point distance: {target_point_distance:.2f} m\n"
        f"Target Speed: {ts:.2f} m/s\n"
        f"Steer: {steer:.2f}\n"
        f"Throttle: {throttle:.2f}\n"
        f"Brake: {brake}"
    )

    # Box turns red when braking
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


    # ----------------------
    # --- LiDAR BEV (CCW) ---
    # ----------------------
    if lidar_bev.ndim == 3:
        lidar_img = np.max(lidar_bev, axis=0)
    else:
        lidar_img = lidar_bev

    lidar_img = np.rot90(lidar_img, k=1)  # rotate anticlockwise

    axes[0, 2].imshow(lidar_img, cmap="gray")
    axes[0, 2].set_title("LiDAR BEV")
    axes[0, 2].axis("off")

    # --------------------
    # --- Depth (invert) ---
    # --------------------
    depth = results.get("pred_depth")
    if depth is not None:
        if hasattr(depth, "detach"):
            depth = depth[0].detach().cpu().numpy()

        # Invert depth safely
        #depth = np.rot90(depth, k=2)

        axes[1, 0].imshow(depth, cmap="plasma")
        axes[1, 0].set_title("Predicted Depth ")
        axes[1, 0].axis("off")
    else:
        axes[1, 0].axis("off")

    # -------------------------
    # --- Camera Semantic ---
    # -------------------------
    sem = results.get("pred_semantic")
    if sem is not None:
        if hasattr(sem, "detach"):
            sem = sem[0].detach().cpu().numpy()

        axes[1, 1].imshow(np.argmax(sem, axis=0))
        axes[1, 1].set_title("Camera Semantic")
        axes[1, 1].axis("off")
    else:
        axes[1, 1].axis("off")

    # -------------------------
    # --- Predicted Waypoints ---
    # -------------------------
    cps = results.get("checkpoints")
    if cps is not None:
        # Rotate trajectory 90° anticlockwise
        cps_rot = np.zeros_like(cps)
        cps_rot[:, 0] = cps[:, 1]   # x' = -y
        cps_rot[:, 1] =  cps[:, 0]   # y' =  x

        axes[1, 2].plot(
            cps_rot[:, 0],
            cps_rot[:, 1],
            "o-",
            label="Predicted Waypoints"
        )
        # axes[1, 2].plot(
        #     route[:, 1],
        #     route[:, 0],
        #     "o-",
        #     label="Ground Truth Waypoints"
        # )
        axes[1, 2].scatter(0, 0, c="red", label="Ego")
        axes[1, 2].axis("equal")
        axes[1, 2].grid(True)
        axes[1, 2].legend()

    plt.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()

    renderer = canvas.get_renderer()
    buf = np.asarray(renderer.buffer_rgba(), dtype=np.uint8)

    h, w, _ = buf.shape
    frame = buf[:, :, :3].copy()  # drop alpha channel

    plt.close(fig)
    return frame


import laspy
# -----------------------------
# Main loop: generate video
# -----------------------------
path = Path("/teamspace/studios/this_studio/idd_processed/20220118103308_seq_10")

def count_files_in_folder(folder_path):
    path = Path(folder_path)
    # Use a list comprehension to filter for files and count them
    files = [p for p in path.iterdir() if p.is_file()]
    return len(files)

total_data = count_files_in_folder(path / "rgb")

for i in tqdm(range(total_data), desc="Generating video"):
    
    image_path = Path(f"{path}/rgb/{i:04d}.jpg")
    lidar_path = Path(f"{path}/lidar/{i:04d}.npy")
    measurement_path = Path(f"{path}/measurements/{i:04d}.json.gz")

    if not image_path.exists() or not lidar_path.exists():
        continue

    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    lidar_points = np.load(str(lidar_path))[:, :3].astype(np.float32)
    # lidar_points = np.zeros((1,3), dtype=np.float32)
    # las = laspy.read(lidar_path)
    # lidar_points = np.vstack((las.x, las.y, las.z)).transpose()
    # with gzip.open(measurement_path, 'rt') as file:
    #     measurement = json.load(file)
    # target_point = measurement["target_point"]
    # speed = measurement["speed"]
    # command = measurement["command"]
    # route = np.array(measurement["route"])
    target_point = [30.0, 0.0]
    speed = 0.0
    command = 4
    route = np.array([[1.0, 0], [2.0, 0], [3.0, 0], [4.0, 0], [5.0, 0], [6.0, 0], [7.0, 0], [8.0, 0], [9.0, 0], [10.0,0]])
    # Preprocess
    target_point_xy = np.asarray(target_point, dtype=np.float32)
    rgb, lidar_bev, tp, vel, cmd = preprocessor.prepare_inputs(
        image, lidar_points, target_point_xy.tolist(), speed, command
    )
    model_target_point_xy = tp.squeeze(0).detach().cpu().numpy()
    with torch.no_grad():
        output = net(rgb, lidar_bev, tp, vel, cmd)
    results = postprocess_model_outputs(
        output,
        config,
        net,
        ego_velocity=vel.item(),
        ego_velocity_tensor=vel
    )

    # Render frame
    # Image.fromarray(np.uint8(lidar_bev.squeeze()*255)).save("lidarbev.png")
    frame_rgb = render_frame_fast(
        image,
        image,
        lidar_bev.squeeze(0).detach().cpu().numpy(),
        results,
        route,
        target_point_xy=model_target_point_xy,
        command=command,
        config=config
    )

    # Initialize video writer once
    if video_writer is None:
        h, w = frame_rgb.shape[:2]
        video_writer = cv2.VideoWriter(
            VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h)
        )

    # Convert RGB -> BGR for OpenCV
    video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

# Finish video
if video_writer is not None:
    video_writer.release()
    print(f"Fast video saved at: {VIDEO_PATH}")
else:
    print("No frames were written. Check that IMAGE_DIR and LIDAR_DIR contain matching files.")