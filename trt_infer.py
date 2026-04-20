import os
import time
import json
import cv2
import numpy as np
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --------------------------------------------------
# GPU / TensorRT Setup
# --------------------------------------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_PATH = "tfpp_fp16.engine"

def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())
_ = torch.tensor([0.0], device="cuda")
engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

# Allocate buffers
inputs, outputs, bindings, host_inputs, host_outputs = [], [], [], {}, {}
stream = cuda.Stream()

# --------------------------------------------------
# Preallocate and keep memory alive
# --------------------------------------------------
bindings = []
inputs, outputs = [], []
host_inputs, host_outputs = {}, {}
device_mem_refs = {}  # keep references to prevent GC
stream = cuda.Stream()

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = context.get_tensor_shape(name)
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    size = trt.volume(shape)

    # Host & Device memory
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    # Keep references alive
    device_mem_refs[name] = device_mem

    bindings.append(int(device_mem))

    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        inputs.append((name, host_mem, device_mem))
        host_inputs[name] = host_mem
    else:
        outputs.append((name, host_mem, device_mem))
        host_outputs[name] = host_mem

# --------------------------------------------------
# Safe TRT inference
# --------------------------------------------------
def trt_infer(rgb, lidar_bev, tp, vel, cmd):

    input_tensors = {
        "rgb": rgb.contiguous(),
        "lidar_bev": lidar_bev.contiguous(),
        "target_point": tp.contiguous(),
        "velocity": vel.contiguous(),
        "command": cmd.contiguous(),
    }

    # D2D copy
    for name, host_mem, device_mem in inputs:
        tensor = input_tensors[name]

        cuda.memcpy_dtod_async(
            device_mem,
            int(tensor.data_ptr()),
            tensor.numel() * tensor.element_size(),
            stream
        )

        context.set_tensor_address(name, int(device_mem))

    # Set output buffers
    for name, host_mem, device_mem in outputs:
        context.set_tensor_address(name, int(device_mem))

    # Execute
    context.execute_async_v3(stream_handle=stream.handle)

    # Copy outputs back (async)
    for name, host_mem, device_mem in outputs:
        cuda.memcpy_dtoh_async(host_mem, device_mem, stream)

    # SINGLE sync at end
    stream.synchronize()

    return {name: host_mem.reshape(context.get_tensor_shape(name))
            for name, host_mem, device_mem in outputs}

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

# --------------------------------------------------
# Load Config + Model (PyTorch)
# --------------------------------------------------
CONFIG_PATH = "models/pretrained_models/all_towns"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print("CUDA Device:", torch.cuda.get_device_name(0))

with open(os.path.join(CONFIG_PATH, "config.json"), "r") as f:
    cfg_dict = json.load(f)
cfg_dict.pop("setting", None)

from .model import LidarCenterNet
from .config import GlobalConfig
from . import transfuser_utils as t_u

config = GlobalConfig()
config.initialize(setting="eval", **cfg_dict)
config.compile = False
config.sync_batch_norm = False

# Load PyTorch model for postprocessing (control PID)
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

# --------------------------------------------------
# Input Preprocessing
# --------------------------------------------------
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
        return torch.from_numpy(features).unsqueeze(0).float().to(self.device)

    def preprocess_target_point(self, target_point):
        tp = torch.from_numpy(np.array(target_point, dtype=np.float32)).unsqueeze(0).to(self.device)
        return tp

    def preprocess_velocity(self, velocity):
        return torch.tensor([[velocity]], dtype=torch.float32).to(self.device)

    def preprocess_command(self, command):
        cmd = torch.from_numpy(t_u.command_to_one_hot(command)).unsqueeze(0).float().to(self.device)
        return cmd

    def prepare_inputs(self, image, lidar_points, target_point, velocity, command):
        rgb = self.preprocess_image(image)
        lidar_bev = self.preprocess_lidar(lidar_points)
        tp = self.preprocess_target_point(target_point)
        vel = self.preprocess_velocity(velocity)
        cmd = self.preprocess_command(command)
        return rgb, lidar_bev, tp, vel, cmd

preprocessor = InputPreprocessor(config, device=DEVICE)

# --------------------------------------------------
# Postprocessing for TensorRT inference (direct controller)
# --------------------------------------------------

def postprocess_model_outputs(output, config, net, ego_velocity, compute_debug_output=False):
    """
    Convert raw TensorRT model outputs into vehicle controls and renderable data.

    Args:
        output: tuple of TRT outputs
        config: configuration object (target speeds, flags)
        net: model wrapper providing control_pid_direct
        ego_velocity: float, current vehicle speed (m/s)

    Returns:
        results: dict with control outputs, waypoints, predictions
    """
    results = {}

    # -------------------------------------------------
    # Helper: convert everything to NumPy safely
    # -------------------------------------------------
    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        # unwrap single-element tuple from TRT if needed
        if isinstance(x, tuple) and len(x) == 1:
            return x[0]
        return x

    # -------------------------------------------------
    # Extract outputs using correct TRT indices
    # -------------------------------------------------
    pred_target_speed = to_numpy(output["pred_target_speed"])       # (1,1,8)
    pred_checkpoint   = to_numpy(output["pred_checkpoint"])         # (1,10,2)
    pred_semantic     = to_numpy(output["pred_semantic"])           # (1,7,384,1024)
    pred_bev_semantic = to_numpy(output["pred_bev_semantic"])       # (1,11,256,256)
    pred_depth        = to_numpy(output["pred_depth"])              # (1,384,1024)

    pred_wp = pred_checkpoint

    # -------------------------------------------------
    # Target speed decoding
    # -------------------------------------------------
    if pred_target_speed is not None:
        logits = pred_target_speed[0, 0]  # shape (8,)
        logits = logits - np.max(logits)  # stability
        exp_x = np.exp(logits)
        probs = exp_x / np.sum(exp_x)
        idx = int(np.argmax(probs))

        if getattr(config, "use_twohot_target_speeds", False):
            if len(config.target_speeds) == logits.shape[0]:
                target_speed = float(
                    np.dot(probs, np.array(config.target_speeds, dtype=np.float32))
                )
            else:
                # fallback if shape mismatch
                target_speed = float(config.target_speeds[idx])
        else:
            target_speed = float(config.target_speeds[idx])
    else:
        probs = None
        target_speed = 0.0

    results["target_speed"] = target_speed
    results["target_speed_probs"] = probs

    # -------------------------------------------------
    # Direct controller PID
    # -------------------------------------------------
    if getattr(config, "inference_direct_controller", False) and pred_checkpoint is not None:
        checkpoints_np = pred_checkpoint[0]  # already NumPy
        steer, throttle, brake = net.control_pid_direct(
            checkpoints_np,
            np.array([[target_speed]], dtype=np.float32),
            np.array([[ego_velocity]], dtype=np.float32),
            ego_vehicle_location=np.array([0.0, 0.0], dtype=np.float32),
            ego_vehicle_rotation=0.0
        )
    else:
        # fallback for legacy PID (should not be used in direct inference)
        pred_wp_tensor = torch.from_numpy(pred_wp) if isinstance(pred_wp, np.ndarray) else pred_wp
        steer, throttle, brake = net.control_pid(
            pred_wp_tensor,
            torch.tensor([ego_velocity], dtype=torch.float32),
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
    # Waypoints / checkpoints
    # -------------------------------------------------
    results["checkpoints"] = pred_checkpoint[0] if pred_checkpoint is not None else None
    results["waypoints"] = pred_wp[0] if pred_wp is not None else None

    # -------------------------------------------------
    # Additional outputs (already NumPy)
    # -------------------------------------------------
    results["pred_semantic"] = pred_semantic
    results["pred_bev_semantic"] = pred_bev_semantic
    results["pred_depth"] = pred_depth

    return results

# --------------------------------------------------
# Rendering function (reuse your existing render_frame_fast)
# --------------------------------------------------
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
# --------------------------------------------------
# Video Inference Loop
# --------------------------------------------------
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Paths
LEFT_ROOT = "idd/primary/d0"
LEFT_IMG_DIR = os.path.join(LEFT_ROOT, "leftCamImgs")
LIDAR_DIR = "idd/lidar/d0"
CSV_CAM = os.path.join(LEFT_ROOT, "train.csv")
CSV_LIDAR = os.path.join(LIDAR_DIR, "timestamp.csv")
IMAGE_EXT = ".jpg"
PAD = 7
VIDEO_PATH = "d0_sequence_synced.mp4"
FPS = 10
max_frames = 50
TARGET_POINT = [10.0, 0.0]
VELOCITY = 5.0
COMMAND = 0
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.9, device=0)

total_inference_time = 0.0
total_frames = 0
max_frames = 50
processed_frames = 0
# Read CSVs
df_cam = pd.read_csv(CSV_CAM)        # columns: timestamp,image_idx,latitude,...
df_lidar = pd.read_csv(CSV_LIDAR)    # columns: frame,datetime

# Convert timestamps to datetime objects for easy comparison
def parse_timestamp(ts):
    # Example: "09-00-31-462000"
    return datetime.strptime(ts, "%H-%M-%S-%f")

df_cam["ts_dt"] = df_cam["timestamp"].apply(parse_timestamp)
df_lidar["ts_dt"] = df_lidar["datetime"].apply(parse_timestamp)

# Sort lidar timestamps for efficient search
lidar_times = df_lidar["ts_dt"].values
lidar_frames = df_lidar["frame"].values

# Video writer (optional)
video_writer = None
total_frames = 0
processed_frames = 0

for _, cam_row in tqdm(df_cam.iterrows(), total=len(df_cam), desc="Generating video"):
    if processed_frames >= max_frames:
        break

    cam_ts = cam_row["ts_dt"]
    img_idx = str(cam_row["image_idx"]).zfill(PAD)
    image_path = os.path.join(LEFT_IMG_DIR, img_idx + IMAGE_EXT)

    if not os.path.exists(image_path):
        continue

    # Find nearest lidar frame
    time_diffs = np.abs(np.array([(lidar_ts - cam_ts).total_seconds() for lidar_ts in lidar_times]))
    nearest_idx = np.argmin(time_diffs)
    lidar_frame = str(lidar_frames[nearest_idx]).zfill(PAD)
    lidar_path = os.path.join(LIDAR_DIR, lidar_frame + ".npy")

    if not os.path.exists(lidar_path):
        continue

    # Load inputs
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    lidar_points = np.load(lidar_path)[:, :3].astype(np.float32)

    # Preprocess
    rgb, lidar_bev, tp, vel, cmd = preprocessor.prepare_inputs(
        image, lidar_points, TARGET_POINT, VELOCITY, COMMAND
    )

    # Run TRT FP16 inference
    start_time = time.time()
    output = trt_infer(rgb, lidar_bev, tp, vel, cmd)
    end_time = time.time()
    inference_time = end_time - start_time
    total_inference_time += inference_time
    total_frames += 1
    processed_frames += 1

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
    print(f"Average inference time per frame: {avg_inference_time:.4f} sec")
    print(f"FP16 TRT Model FPS: {model_fps:.2f}")

