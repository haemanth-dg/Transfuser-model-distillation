import argparse
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    import laspy
except ImportError:
    laspy = None

try:
    from . import transfuser_utils as t_u
    from .config import GlobalConfig
    from .model import LidarCenterNet
except ImportError:
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from model_nocarla import transfuser_utils as t_u
    from model_nocarla.config import GlobalConfig
    from model_nocarla.model import LidarCenterNet

SCENE_PATH = Path("/teamspace/studios/this_studio/idd_processed/20220118103308_seq_10")
CONFIG_PATH = Path("/teamspace/studios/this_studio/models/pretrained_models/all_towns")
CHECKPOINT = Path("/teamspace/studios/this_studio/models/pretrained_models/all_towns/model_final_merged.pth")
OUTPUT_PATH = Path("videos/bbox_debug_idd.mp4")
FPS = 10
HIDE_GT = False

CLASS_TO_ID = {
    "car": 0,
    "walker": 1,
    "traffic_light": 2,
    "stop_sign": 3,
    "emergency_vehicle": 4,
}

# OpenCV uses BGR
COLOR_RAW = (255, 160, 0)
COLOR_FILTERED = (0, 255, 255)
COLOR_GT = (0, 0, 255)
COLOR_EGO = (0, 255, 0)
COLOR_TEXT = (230, 230, 230)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device_str):
    if device_str is None or str(device_str).lower() == "auto":
        return get_default_device()
    return torch.device(device_str)


def load_config(config_path):
    config_path = Path(config_path)
    config_json = config_path / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"Missing config file: {config_json}")

    with open(config_json, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    cfg_dict.pop("setting", None)

    config = GlobalConfig()
    config.initialize(setting="eval", **cfg_dict)
    config.compile = False
    config.sync_batch_norm = False
    config.detect_boxes = 1
    return config


def resolve_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        return checkpoint_path
    if checkpoint_path.is_dir():
        candidates = sorted(checkpoint_path.glob("model*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No model*.pth found in {checkpoint_path}")
        return candidates[0]
    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")


def load_model(config, checkpoint_path, device):
    checkpoint = resolve_checkpoint(checkpoint_path)
    net = LidarCenterNet(config).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net, checkpoint


def load_measurement(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def resolve_lidar_path(scene_path, frame_id):
    lidar_dir = Path(scene_path) / "lidar"
    for ext in (".npy", ".laz"):
        candidate = lidar_dir / f"{frame_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_lidar_points(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        pts = np.load(path)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Invalid LiDAR shape in {path}: {pts.shape}")
        return pts[:, :3].astype(np.float32)
    if suffix == ".laz":
        if laspy is None:
            raise ImportError("laspy is required for .laz LiDAR files. Install with: pip install laspy")
        laz = laspy.read(path)
        return np.column_stack((laz.x, laz.y, laz.z)).astype(np.float32)
    raise ValueError(f"Unsupported LiDAR format: {path}")


def lidar_to_histogram_features(lidar, config):
    lidar = lidar[(lidar[:, 0] >= config.min_x) & (lidar[:, 0] < config.max_x)]
    lidar = lidar[(lidar[:, 1] >= config.min_y) & (lidar[:, 1] < config.max_y)]

    if config.use_ground_plane:
        below = lidar[lidar[:, 2] <= config.lidar_split_height]
        above = lidar[lidar[:, 2] > config.lidar_split_height]

        below_px_x = ((below[:, 0] - config.min_x) * config.pixels_per_meter).astype(np.int32)
        below_px_y = ((below[:, 1] - config.min_y) * config.pixels_per_meter).astype(np.int32)
        above_px_x = ((above[:, 0] - config.min_x) * config.pixels_per_meter).astype(np.int32)
        above_px_y = ((above[:, 1] - config.min_y) * config.pixels_per_meter).astype(np.int32)

        below_hist = np.zeros((config.lidar_resolution_height, config.lidar_resolution_width), dtype=np.float32)
        above_hist = np.zeros((config.lidar_resolution_height, config.lidar_resolution_width), dtype=np.float32)

        below_px_x = np.clip(below_px_x, 0, config.lidar_resolution_width - 1)
        below_px_y = np.clip(below_px_y, 0, config.lidar_resolution_height - 1)
        above_px_x = np.clip(above_px_x, 0, config.lidar_resolution_width - 1)
        above_px_y = np.clip(above_px_y, 0, config.lidar_resolution_height - 1)

        np.add.at(below_hist, (below_px_x, below_px_y), 1)
        np.add.at(above_hist, (above_px_x, above_px_y), 1)

        below_hist = np.clip(below_hist / config.hist_max_per_pixel, 0, 1).T
        above_hist = np.clip(above_hist / config.hist_max_per_pixel, 0, 1).T
        return np.stack([below_hist, above_hist], axis=0)

    px_x = ((lidar[:, 0] - config.min_x) * config.pixels_per_meter).astype(np.int32)
    px_y = ((lidar[:, 1] - config.min_y) * config.pixels_per_meter).astype(np.int32)

    hist = np.zeros((config.lidar_resolution_height, config.lidar_resolution_width), dtype=np.float32)
    px_x = np.clip(px_x, 0, config.lidar_resolution_width - 1)
    px_y = np.clip(px_y, 0, config.lidar_resolution_height - 1)
    np.add.at(hist, (px_x, px_y), 1)

    hist = np.clip(hist / config.hist_max_per_pixel, 0, 1).T
    return hist[np.newaxis, :, :]


def preprocess_inputs(config, device, rgb_bgr, lidar_points, measurement):
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (config.camera_width, config.camera_height))
    rgb = t_u.crop_array(config, rgb)
    rgb_tensor = torch.from_numpy(np.transpose(rgb.astype(np.float32), (2, 0, 1))).unsqueeze(0).to(device)

    lidar_bev = lidar_to_histogram_features(lidar_points, config)
    if lidar_bev.ndim == 2:
        lidar_bev = lidar_bev[None, :, :]
    lidar_tensor = torch.from_numpy(lidar_bev).unsqueeze(0).float().to(device)

    target_point = measurement.get("target_point", [0.0, 0.0])
    speed = float(measurement.get("speed", 0.0))
    command = int(measurement.get("command", 4))

    target_point_tensor = torch.from_numpy(np.array(target_point, dtype=np.float32)).unsqueeze(0).to(device)
    velocity_tensor = torch.tensor([[speed]], dtype=torch.float32).to(device)
    command_tensor = torch.from_numpy(t_u.command_to_one_hot(command)).unsqueeze(0).float().to(device)

    return rgb_tensor, lidar_tensor, target_point_tensor, velocity_tensor, command_tensor, lidar_bev


def extract_bbox_outputs(net, config, pred_bb_features, raw_conf_floor):
    raw = net.head.get_bboxes(
        pred_bb_features[0],
        pred_bb_features[1],
        pred_bb_features[2],
        pred_bb_features[3],
        pred_bb_features[4],
        pred_bb_features[5],
        pred_bb_features[6],
    )[0]

    raw_np = raw.detach().cpu().numpy()
    raw_np = raw_np[raw_np[:, -1] >= raw_conf_floor]

    filtered_np = raw_np[raw_np[:, -1] > config.bb_confidence_threshold]

    filtered_metric = []
    for box in filtered_np:
        metric_box = t_u.bb_image_to_vehicle_system(
            box.copy(),
            config.pixels_per_meter,
            config.min_x,
            config.min_y,
        )
        filtered_metric.append(metric_box)

    return raw_np, filtered_metric


def iter_frame_ids(scene_path, start_index, count, image_ext):
    rgb_dir = Path(scene_path) / "rgb"
    frames = sorted(rgb_dir.glob(f"*{image_ext}"))
    if not frames:
        raise FileNotFoundError(f"No frames with extension {image_ext} in {rgb_dir}")

    if start_index < 0 or start_index >= len(frames):
        raise ValueError(f"start-index {start_index} out of range for {len(frames)} frames")

    end_index = min(start_index + count, len(frames)) if count > 0 else len(frames)
    return [p.stem for p in frames[start_index:end_index]]


def load_gt_boxes(scene_path, frame_id):
    boxes_path = Path(scene_path) / "boxes" / f"{frame_id}.json.gz"
    if not boxes_path.exists():
        return []

    try:
        with gzip.open(boxes_path, "rt", encoding="utf-8") as f:
            boxes = json.load(f)
    except OSError:
        return []

    gt_metric = []
    for box in boxes:
        cls = box.get("class", "")
        if cls not in CLASS_TO_ID:
            continue

        position = box.get("position", [0.0, 0.0, 0.0])
        extent = box.get("extent", [0.0, 0.0, 0.0])
        yaw = float(box.get("yaw", 0.0))
        speed = float(box.get("speed", 0.0))
        brake = float(box.get("brake", 0.0)) if not np.isnan(box.get("brake", 0.0)) else 0.0

        # [x, y, w, h, yaw, speed, brake, class, confidence]
        gt_metric.append(
            np.array(
                [
                    float(position[0]),
                    float(position[1]),
                    float(extent[0]),
                    float(extent[1]),
                    yaw,
                    speed,
                    brake,
                    float(CLASS_TO_ID[cls]),
                    1.0,
                ],
                dtype=np.float32,
            )
        )

    return gt_metric


def box_corners_xyxy(center_x, center_y, half_w, half_h, yaw):
    corners = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float32,
    )
    rot = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float32,
    )
    return (rot @ corners.T).T + np.array([center_x, center_y], dtype=np.float32)


def vehicle_to_pixel(x, y, config, size):
    px = int((y - config.min_y) / (config.max_y - config.min_y) * (size - 1))
    py = int((config.max_x - x) / (config.max_x - config.min_x) * (size - 1))
    return px, py


def draw_vehicle_box(canvas, box, config, color, size, thickness=2):
    corners_xy = box_corners_xyxy(box[0], box[1], box[2], box[3], box[4])
    corners_px = [vehicle_to_pixel(cx, cy, config, size) for cx, cy in corners_xy]
    corners_px = np.array(corners_px, dtype=np.int32)
    cv2.polylines(canvas, [corners_px], isClosed=True, color=color, thickness=thickness)


def draw_raw_box(canvas, box, size, thickness=1):
    # raw boxes are in BEV image coordinates where the model predicts on lidar-resolution space
    scale_x = size / 256.0
    scale_y = size / 256.0

    center_x = float(box[1]) * scale_x
    center_y = float(box[0]) * scale_y
    half_w = float(box[3]) * scale_x
    half_h = float(box[2]) * scale_y
    yaw = float(box[4])

    corners = box_corners_xyxy(center_x, center_y, half_w, half_h, yaw)
    corners_int = np.round(corners).astype(np.int32)
    cv2.polylines(canvas, [corners_int], isClosed=True, color=COLOR_RAW, thickness=thickness)


def draw_lidar_points_metric(canvas, lidar_points, config, size):
    x = lidar_points[:, 0]
    y = lidar_points[:, 1]

    valid = (
        (x >= config.min_x)
        & (x <= config.max_x)
        & (y >= config.min_y)
        & (y <= config.max_y)
    )

    x = x[valid]
    y = y[valid]

    px = ((y - config.min_y) / (config.max_y - config.min_y) * (size - 1)).astype(np.int32)
    py = ((config.max_x - x) / (config.max_x - config.min_x) * (size - 1)).astype(np.int32)

    canvas[py, px] = np.maximum(canvas[py, px], np.array([220, 220, 220], dtype=np.uint8))


def build_raw_panel(lidar_bev, raw_boxes, panel_size):
    if lidar_bev.ndim == 3:
        lidar_img = np.max(lidar_bev, axis=0)
    else:
        lidar_img = lidar_bev

    raw_base = (np.clip(lidar_img, 0.0, 1.0) ** 0.5 * 255.0).astype(np.uint8)
    raw_base = cv2.resize(raw_base, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST)
    panel = cv2.cvtColor(raw_base, cv2.COLOR_GRAY2BGR)

    for box in raw_boxes:
        draw_raw_box(panel, box, panel_size, thickness=1)

    cv2.putText(panel, "Raw Decoded Boxes (BEV image space)", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    cv2.putText(panel, f"count={len(raw_boxes)}", (14, panel_size - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RAW, 2)
    return panel


def build_metric_panel(lidar_points, filtered_metric, gt_metric, config, panel_size):
    panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    draw_lidar_points_metric(panel, lidar_points, config, panel_size)

    for box in filtered_metric:
        draw_vehicle_box(panel, box, config, color=COLOR_FILTERED, size=panel_size, thickness=2)

    for box in gt_metric:
        draw_vehicle_box(panel, box, config, color=COLOR_GT, size=panel_size, thickness=2)

    ego_px, ego_py = vehicle_to_pixel(0.0, 0.0, config, panel_size)
    cv2.circle(panel, (ego_px, ego_py), 4, COLOR_EGO, -1)

    cv2.putText(panel, "Filtered Pred + GT (vehicle metric space)", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    cv2.putText(panel, f"pred={len(filtered_metric)} gt={len(gt_metric)}", (14, panel_size - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FILTERED, 2)
    return panel


def render_debug_frame(lidar_bev, lidar_points, raw_boxes, filtered_metric, gt_metric, config, panel_size=800):
    left = build_raw_panel(lidar_bev, raw_boxes, panel_size)
    right = build_metric_panel(lidar_points, filtered_metric, gt_metric, config, panel_size)

    frame = np.hstack([left, right])
    cv2.putText(frame, "Legend: raw=orange filtered=yellow gt=red ego=green", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT, 2)
    return frame


def make_parser():

    parser = argparse.ArgumentParser(description="BBox-only debug video generator")
    parser.add_argument("--scene-path", type=Path, default=SCENE_PATH, help="Scene folder with rgb/lidar/measurements")
    parser.add_argument("--config-path", type=Path, default=CONFIG_PATH, help="Folder containing config.json")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT, help="Checkpoint file or directory with model*.pth")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output MP4 path")
    parser.add_argument("--start-index", type=int, default=0, help="Start frame index in sorted rgb list")
    parser.add_argument("--count", type=int, default=0, help="Number of frames to process; <=0 means until end")
    parser.add_argument("--fps", type=int, default=FPS, help="Output video FPS")
    parser.add_argument("--raw-conf-floor", type=float, default=0.05, help="Min confidence for raw decoded boxes")
    parser.add_argument("--panel-size", type=int, default=800, help="Square panel size for each side")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--image-ext", type=str, default=".jpg", help="RGB image extension in scene/rgb")
    parser.add_argument("--jsonl-path", type=Path, default=None, help="Optional path to write per-frame bbox metadata")
    parser.add_argument("--hide-raw", action="store_true", help="Disable raw decoded box overlay")
    parser.add_argument("--hide-filtered", action="store_true", help="Disable filtered prediction overlay")
    parser.add_argument("--hide-gt", action="store_true", default=HIDE_GT, help="Disable GT box overlay")
    return parser


def validate_scene(scene_path):
    # `measurements` is optional for debug runs — allow its absence and
    # proceed using dummy measurement values per-frame.
    required = ["rgb", "lidar"]
    missing = [name for name in required if not (scene_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Scene is missing required folders: {missing}")

    if not (scene_path / "measurements").exists():
        print(f"Warning: 'measurements' folder not found in scene {scene_path}; using dummy measurements where needed.")


def main():
    args = make_parser().parse_args()

    scene_path = args.scene_path
    validate_scene(scene_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    config = load_config(args.config_path)
    net, checkpoint = load_model(config, args.checkpoint, device)

    frame_ids = iter_frame_ids(scene_path, args.start_index, args.count, args.image_ext)
    print(f"Model checkpoint: {checkpoint}")
    print(f"Device: {device}")
    print(f"Frames selected: {len(frame_ids)}")

    writer = None
    jsonl_file = None
    if args.jsonl_path is not None:
        args.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(args.jsonl_path, "w", encoding="utf-8")

    skipped = 0

    try:
        for frame_id in tqdm(frame_ids, desc="bbox_debug"):
            rgb_path = scene_path / "rgb" / f"{frame_id}{args.image_ext}"
            measurement_path = scene_path / "measurements" / f"{frame_id}.json.gz"
            lidar_path = resolve_lidar_path(scene_path, frame_id)

            # Only skip if RGB or LiDAR missing; allow missing measurements and use dummies
            if not rgb_path.exists() or lidar_path is None:
                skipped += 1
                continue

            rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                skipped += 1
                continue

            # Load measurement if available, otherwise fall back to safe defaults
            if measurement_path.exists():
                try:
                    measurement = load_measurement(measurement_path)
                except Exception as e:
                    print(f"Warning: failed to load measurement {measurement_path}: {e}")
                    measurement = {"target_point": [0.0, 0.0], "speed": 0.0, "command": 4}
            else:
                measurement = {"target_point": [0.0, 0.0], "speed": 0.0, "command": 4}

            lidar_points = load_lidar_points(lidar_path)

            rgb, lidar_bev, tp, vel, cmd, lidar_bev_np = preprocess_inputs(
                config,
                device,
                rgb_bgr,
                lidar_points,
                measurement,
            )

            with torch.no_grad():
                output = net(rgb, lidar_bev, tp, vel, cmd)

            pred_bb_features = output[6]
            raw_boxes, filtered_metric = extract_bbox_outputs(net, config, pred_bb_features, args.raw_conf_floor)
            gt_metric = load_gt_boxes(scene_path, frame_id)

            if args.hide_raw:
                raw_boxes = []
            if args.hide_filtered:
                filtered_metric = []
            if args.hide_gt:
                gt_metric = []

            frame = render_debug_frame(
                lidar_bev_np,
                lidar_points,
                raw_boxes,
                filtered_metric,
                gt_metric,
                config,
                panel_size=args.panel_size,
            )

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (w, h),
                )

            writer.write(frame)

            if jsonl_file is not None:
                meta = {
                    "frame_id": frame_id,
                    "raw_count": int(len(raw_boxes)),
                    "filtered_count": int(len(filtered_metric)),
                    "gt_count": int(len(gt_metric)),
                    "bb_confidence_threshold": float(config.bb_confidence_threshold),
                    "raw_conf_floor": float(args.raw_conf_floor),
                }
                jsonl_file.write(json.dumps(meta) + "\n")

    finally:
        if writer is not None:
            writer.release()
        if jsonl_file is not None:
            jsonl_file.close()

    print(f"Saved video: {args.output}")
    if args.jsonl_path is not None:
        print(f"Saved metadata: {args.jsonl_path}")
    print(f"Processed frames: {len(frame_ids) - skipped}")
    print(f"Skipped frames: {skipped}")


if __name__ == "__main__":
    main()
