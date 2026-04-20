import argparse
import gzip
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
	import laspy
except ImportError:
	laspy = None

try:
	from . import transfuser_utils as t_u
	from .config import GlobalConfig
	from .student_model import StudentNet
except ImportError:
	from model_nocarla import transfuser_utils as t_u
	from model_nocarla.config import GlobalConfig
	from model_nocarla.student_model import StudentNet


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "carla_garage" / "output" / "final_merged_model" / "config.json"
DEFAULT_STUDENT_CKPT = REPO_ROOT / "model_distillation" / "output_2" / "student_best.pth"

DEFAULT_SCENE_PATH = "/home/haemanth/Transfuser++/nuplan/nuplan_workspace/nuplan_data/group_0/validation_2021.06.03.12.02.06_veh-35_00233_00609"
DEFAULT_VIDEO_PATH = "/home/haemanth/Transfuser++/carla_garage/Videos/nuplan_route_stud.mp4"
DEFAULT_LOGO_PATH = Path(__file__).resolve().parents[2] / "logo.jpeg"
print(f"Default logo path: {DEFAULT_LOGO_PATH}")
DEFAULT_FPS = 10
DEFAULT_DEVICE = "auto"

HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_TEXT_SCALE = 0.42
HUD_TEXT_THICKNESS = 1
POPPINS_FONT = None
POPPINS_FONT_SIZE = 13
SIDE_PANEL_FONT_SIZE = 11

def load_poppins_font(bold=False, size=None):
	global POPPINS_FONT
	if size is None:
		size = POPPINS_FONT_SIZE
	try:
		font_paths = []
		if bold:
			font_paths = [
				"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
				"/System/Library/Fonts/Arial Bold.ttf",
				"C:\\Windows\\Fonts\\arialbd.ttf",
			]
		else:
			font_paths = [
				"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
				"/System/Library/Fonts/Arial.ttf",
				"C:\\Windows\\Fonts\\arial.ttf",
			]
		for path in font_paths:
			if os.path.exists(path):
				return ImageFont.truetype(path, size)
		return ImageFont.load_default()
	except:
		return ImageFont.load_default()

def draw_text_pil(frame, text, x, y, color, font_obj=None, bold=False):
	"""Draw text using PIL with better font support."""
	if font_obj is None:
		font_obj = load_poppins_font(bold=bold)
	pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	draw = ImageDraw.Draw(pil_img)
	draw.text((int(x), int(y)), text, font=font_obj, fill=color)
	cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR, dst=frame)

# Color Palette
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW_GREEN = (11, 248, 207)
COLOR_PINK_RED = (34, 255, 87)
COLOR_VERY_DARK = (17, 17, 17)
COLOR_BRIGHT_GREEN = (164, 255, 0)
COLOR_DARK_GRAY = (30, 30, 30)

def draw_rounded_rect(image, x1, y1, x2, y2, color, radius=8, alpha=1.0):
	"""Draw a rounded rectangle with specified radius."""
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
	radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
	
	overlay = image.copy()
	
	cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
	cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
	
	cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
	cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
	cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
	cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
	
	cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def draw_gradient_fill(image, x1, y1, x2, y2, color_top, color_bottom, alpha=1.0):
	"""Draw a vertical gradient fill."""
	x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
	height = y2 - y1
	
	if height > 0:
		for i in range(height):
			ratio = i / max(1, height)
			r = int(color_top[0] * (1 - ratio) + color_bottom[0] * ratio)
			g = int(color_top[1] * (1 - ratio) + color_bottom[1] * ratio)
			b = int(color_top[2] * (1 - ratio) + color_bottom[2] * ratio)
			
			overlay = image.copy()
			cv2.line(overlay, (x1, y1 + i), (x2, y1 + i), (b, g, r), thickness=x2 - x1 + 1)
			cv2.addWeighted(overlay, alpha / max(1, height), image, 1.0 - alpha / max(1, height), 0.0, dst=image)


def get_default_device():
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def resolve_device(device=None):
	if device is None or str(device).lower() == "auto":
		return get_default_device()
	return torch.device(device)


def get_model_device(net):
	try:
		return next(net.parameters()).device
	except StopIteration:
		return torch.device("cpu")


def _load_json(path):
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def _extract_state_dict(ckpt_obj):
	if not isinstance(ckpt_obj, dict):
		return ckpt_obj

	for key in ("state_dict", "model_state_dict", "model", "net"):
		if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
			return ckpt_obj[key]

	return ckpt_obj


def _strip_prefix_if_present(state_dict, prefix):
	if not isinstance(state_dict, dict) or not state_dict:
		return state_dict
	if not all(k.startswith(prefix) for k in state_dict.keys()):
		return state_dict
	return {k[len(prefix):]: v for k, v in state_dict.items()}


def _load_student_checkpoint(net, ckpt_path, device):
	print(f"Loading student checkpoint: {ckpt_path}")
	ckpt_obj = torch.load(ckpt_path, map_location=device)
	state_dict = _extract_state_dict(ckpt_obj)
	state_dict = _strip_prefix_if_present(state_dict, "module.")

	model_state = net.state_dict()
	filtered_state_dict = {}
	skipped_shape = []

	for key, value in state_dict.items():
		if key not in model_state:
			continue
		if model_state[key].shape != value.shape:
			skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
			continue
		filtered_state_dict[key] = value

	missing, unexpected = net.load_state_dict(filtered_state_dict, strict=False)
	print(f"Loaded {len(filtered_state_dict)}/{len(model_state)} tensors from {Path(ckpt_path).name}")
	print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
	if skipped_shape:
		print(f"Skipped shape-mismatched keys: {len(skipped_shape)}")
		for name, src_shape, dst_shape in skipped_shape[:10]:
			print(f"  {name}: ckpt={src_shape}, model={dst_shape}")
		if len(skipped_shape) > 10:
			print("  ...")


def load_config_and_model(config_path, student_ckpt_path, device=None):
	device = resolve_device(device)

	cfg_dict = _load_json(config_path)

	cfg_dict.pop("setting", None)

	config = GlobalConfig()
	config.initialize(setting="eval", **cfg_dict)
	config.compile = False
	config.sync_batch_norm = False
	config.use_semantic = False
	config.use_bev_semantic = False
	config.use_depth = False
	config.detect_boxes = False
	config.use_wp_gru = False
	config.lidar_seq_len = 1
	config.augment = False
	config.use_ground_plane = False
	config.only_perception = False
	config.inference_direct_controller = True

	net = StudentNet(config, use_kd_projectors=True).to(device)

	_load_student_checkpoint(net, student_ckpt_path, device)

	net.eval()

	print(f"Student model loaded successfully on {device}")
	return config, net


def lidar_to_histogram_features(lidar, config):
	lidar = lidar[(lidar[:, 0] >= config.min_x) & (lidar[:, 0] < config.max_x)]
	lidar = lidar[(lidar[:, 1] >= config.min_y) & (lidar[:, 1] < config.max_y)]

	if config.use_ground_plane:
		below = lidar[lidar[:, 2] <= config.lidar_split_height]
		above = lidar[lidar[:, 2] > config.lidar_split_height]

		below_pixels_x = ((below[:, 0] - config.min_x) * config.pixels_per_meter).astype(np.int32)
		below_pixels_y = ((below[:, 1] - config.min_y) * config.pixels_per_meter).astype(np.int32)
		above_pixels_x = ((above[:, 0] - config.min_x) * config.pixels_per_meter).astype(np.int32)
		above_pixels_y = ((above[:, 1] - config.min_y) * config.pixels_per_meter).astype(np.int32)

		below_histogram = np.zeros((config.lidar_resolution_height, config.lidar_resolution_width), dtype=np.float32)
		above_histogram = np.zeros((config.lidar_resolution_height, config.lidar_resolution_width), dtype=np.float32)

		below_pixels_x = np.clip(below_pixels_x, 0, config.lidar_resolution_width - 1)
		below_pixels_y = np.clip(below_pixels_y, 0, config.lidar_resolution_height - 1)
		above_pixels_x = np.clip(above_pixels_x, 0, config.lidar_resolution_width - 1)
		above_pixels_y = np.clip(above_pixels_y, 0, config.lidar_resolution_height - 1)

		np.add.at(below_histogram, (below_pixels_x, below_pixels_y), 1)
		np.add.at(above_histogram, (above_pixels_x, above_pixels_y), 1)

		below_histogram = np.clip(below_histogram / config.hist_max_per_pixel, 0, 1).T
		above_histogram = np.clip(above_histogram / config.hist_max_per_pixel, 0, 1).T
		return np.stack([below_histogram, above_histogram], axis=0)

	pixels_x = ((lidar[:, 0] - config.min_x) * config.pixels_per_meter).astype(np.int32)
	pixels_y = ((lidar[:, 1] - config.min_y) * config.pixels_per_meter).astype(np.int32)

	histogram = np.zeros((config.lidar_resolution_height, config.lidar_resolution_width), dtype=np.float32)
	pixels_x = np.clip(pixels_x, 0, config.lidar_resolution_width - 1)
	pixels_y = np.clip(pixels_y, 0, config.lidar_resolution_height - 1)
	np.add.at(histogram, (pixels_x, pixels_y), 1)

	histogram = np.clip(histogram / config.hist_max_per_pixel, 0, 1).T
	return histogram[np.newaxis, :, :]


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
		return torch.from_numpy(np.array(target_point, dtype=np.float32)).unsqueeze(0).to(self.device)

	def preprocess_velocity(self, velocity):
		return torch.tensor([[velocity]], dtype=torch.float32).to(self.device)

	def preprocess_command(self, command):
		return torch.from_numpy(t_u.command_to_one_hot(command)).unsqueeze(0).float().to(self.device)

	def prepare_inputs(self, image, lidar_points, target_point, velocity, command):
		rgb = self.preprocess_image(image)
		lidar_bev = self.preprocess_lidar(lidar_points)
		target_point_tensor = self.preprocess_target_point(target_point)
		velocity_tensor = self.preprocess_velocity(velocity)
		command_tensor = self.preprocess_command(command)
		return rgb, lidar_bev, target_point_tensor, velocity_tensor, command_tensor


def postprocess_model_outputs(output, config, net, ego_velocity):
	results = {}

	pred_target_speed = output[1]
	pred_checkpoint = output[2]
	pred_semantic = output[3]
	pred_bev_semantic = output[4]
	pred_depth = output[5]
	pred_bb_features = output[6]

	if getattr(config, "detect_boxes", False):
		pred_bounding_box = net.convert_features_to_bb_metric(pred_bb_features)
	else:
		pred_bounding_box = None

	if pred_target_speed is not None:
		probs = F.softmax(pred_target_speed, dim=1)[0].cpu().numpy()
		idx = int(np.argmax(probs))
		if getattr(config, "use_twohot_target_speeds", False):
			target_speed = sum(prob * speed for prob, speed in zip(probs, config.target_speeds))
		else:
			target_speed = config.target_speeds[idx]
		target_speed = float(target_speed)
	else:
		probs = None
		target_speed = 0.0

	if getattr(config, "inference_direct_controller", False) and pred_checkpoint is not None:
		checkpoints_np = pred_checkpoint[0].cpu().numpy()
		ego_velocity_tensor = torch.tensor([[ego_velocity]], dtype=torch.float32, device=pred_checkpoint.device)

		steer, throttle, brake = net.control_pid_direct(
			checkpoints_np,
			target_speed,
			ego_velocity_tensor,
			ego_vehicle_location=np.array([0.0, 0.0], dtype=np.float32),
			ego_vehicle_rotation=0.0,
		)
	else:
		velocity_device = pred_checkpoint.device if pred_checkpoint is not None else get_model_device(net)
		ego_velocity_tensor = torch.tensor([[ego_velocity]], dtype=torch.float32, device=velocity_device)
		steer, throttle, brake = net.control_pid(pred_checkpoint, ego_velocity_tensor, tuned_aim_distance=False)

	if ego_velocity < 0.1:
		throttle = max(getattr(config, "creep_throttle", 0.1), throttle)
		brake = False

	results["pred_bb"] = pred_bounding_box
	results["target_speed"] = target_speed
	results["target_speed_probs"] = probs
	results["steer"] = float(steer)
	results["throttle"] = float(throttle)
	results["brake"] = bool(brake)
	results["checkpoints"] = pred_checkpoint[0].cpu().numpy() if pred_checkpoint is not None else None
	results["pred_semantic"] = pred_semantic
	results["pred_bev_semantic"] = pred_bev_semantic
	results["pred_depth"] = pred_depth
	return results


def load_measurement(path):
	with gzip.open(path, "rt", encoding="utf-8") as file:
		return json.load(file)


def resolve_lidar_path(scene_path, frame_id):
	lidar_dir = scene_path / "lidar"
	for extension in (".npy", ".laz"):
		lidar_path = lidar_dir / f"{frame_id}{extension}"
		if lidar_path.exists():
			return lidar_path
	return None


def load_lidar_points(path):
	suffix = path.suffix.lower()

	if suffix == ".npy":
		lidar_raw = np.load(path)
		if lidar_raw.ndim != 2 or lidar_raw.shape[1] < 3:
			raise ValueError(f"Invalid LiDAR shape in {path}: {lidar_raw.shape}")
		return lidar_raw[:, :3].astype(np.float32)

	if suffix == ".laz":
		if laspy is None:
			raise ImportError("laspy is required to read .laz LiDAR files")
		lidar_raw = laspy.read(path)
		return np.column_stack((lidar_raw.x, lidar_raw.y, lidar_raw.z)).astype(np.float32)

	raise ValueError(f"Unsupported LiDAR format: {path}")


def normalize_route(route, target_length=None):
	route_array = np.asarray(route, dtype=np.float32)
	if route_array.ndim != 2 or route_array.shape[1] != 2:
		raise ValueError(f"Expected route with shape (N, 2), got {route_array.shape}")

	if target_length is None:
		return route_array

	if len(route_array) >= target_length:
		return route_array[:target_length]

	padding = np.repeat(route_array[-1][None, :], target_length - len(route_array), axis=0)
	return np.vstack([route_array, padding])


def command_label(command):
	labels = {
		-1: "Lane Follow",
		0: "Follow",
		1: "Left",
		2: "Right",
		3: "Straight",
		4: "Lane Follow",
		5: "Change Left",
		6: "Change Right",
	}
	return labels.get(int(command), f"Command {command}")


def draw_filled_rect(image, x1, y1, x2, y2, color, alpha=1.0):
	x1 = max(0, min(image.shape[1], int(x1)))
	x2 = max(0, min(image.shape[1], int(x2)))
	y1 = max(0, min(image.shape[0], int(y1)))
	y2 = max(0, min(image.shape[0], int(y2)))
	if x2 <= x1 or y2 <= y1:
		return

	if alpha >= 0.999:
		image[y1:y2, x1:x2] = color
		return

	overlay = image.copy()
	overlay[y1:y2, x1:x2] = color
	cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def blend_icon_rgba(dst, icon, ox, oy):
	if icon is None or icon.size == 0:
		return

	if icon.shape[2] == 3:
		alpha_channel = np.full(icon.shape[:2] + (1,), 255, dtype=np.uint8)
		icon = np.concatenate([icon, alpha_channel], axis=2)

	ih, iw = icon.shape[:2]
	h, w = dst.shape[:2]
	x0 = max(0, ox)
	y0 = max(0, oy)
	x1 = min(w, ox + iw)
	y1 = min(h, oy + ih)
	if x1 <= x0 or y1 <= y0:
		return

	sx0 = x0 - ox
	sy0 = y0 - oy
	sx1 = sx0 + (x1 - x0)
	sy1 = sy0 + (y1 - y0)

	src = icon[sy0:sy1, sx0:sx1].astype(np.float32)
	alpha = (src[:, :, 3:4] / 255.0).astype(np.float32)
	fg = src[:, :, :3]
	bg = dst[y0:y1, x0:x1].astype(np.float32)
	out = fg * alpha + bg * (1.0 - alpha)
	dst[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)


def load_logo_rgba(path, target_height):
	if path is None or not path.exists():
		return None

	logo = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
	if logo is None or logo.size == 0:
		return None

	if logo.ndim == 2:
		logo_rgb = cv2.cvtColor(logo, cv2.COLOR_GRAY2RGB)
		alpha_channel = np.full(logo_rgb.shape[:2], 255, dtype=np.uint8)
		logo = np.dstack([logo_rgb, alpha_channel])
	elif logo.shape[2] == 3:
		logo_rgb = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
		alpha_channel = np.where(gray > 252, 0, 255).astype(np.uint8)
		alpha_channel = cv2.GaussianBlur(alpha_channel, (3, 3), 0)
		if np.count_nonzero(alpha_channel > 20) < int(alpha_channel.size * 0.02):
			alpha_channel[:] = 255
		logo = np.dstack([logo_rgb, alpha_channel])
	else:
		logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2RGBA)

	scale = max(0.05, target_height / float(logo.shape[0]))
	new_w = max(1, int(logo.shape[1] * scale))
	new_h = max(1, int(logo.shape[0] * scale))
	return cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_metric_card(frame, x1, y1, x2, y2, title, value, accent):
	draw_rounded_rect(frame, x1, y1, x2, y2, COLOR_DARK_GRAY, radius=5, alpha=0.92)
	cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), accent, 1, cv2.LINE_AA)
	font_bold = load_poppins_font(bold=True)
	font = load_poppins_font(bold=False)
	draw_text_pil(frame, title, x1 + 8, y1 + 14, COLOR_WHITE, font_bold, bold=True)
	draw_text_pil(frame, value, x1 + 8, y2 - 18, accent, font_bold, bold=True)


def draw_top_hud(frame, measurement, results, frame_idx):
	h, w = frame.shape[:2]
	hud_h = max(34, int(h * 0.052))
	pad = 10
	outer_margin = max(4, int(min(h, w) * 0.006))

	speed = float(measurement.get("speed", 0.0)) * 3.6
	target_speed = float(results.get("target_speed", 0.0)) * 3.6
	command = command_label(measurement.get("command", 0))

	# Full-length top bar without gradient.
	x1 = outer_margin
	x2 = w - outer_margin
	draw_rounded_rect(frame, x1, 0, x2, hud_h, COLOR_DARK_GRAY, radius=8, alpha=0.90)
	cv2.rectangle(frame, (int(x1), 0), (int(x2), int(hud_h)), COLOR_BRIGHT_GREEN, 1, cv2.LINE_AA)

	# Single centered row: Speed | Target Speed | Command
	y_text = max(8, int((hud_h - POPPINS_FONT_SIZE) // 2) - 1)
	font_bold = load_poppins_font(bold=True)

	items = [
		("Speed", f"{speed:.1f} km/h", COLOR_BRIGHT_GREEN),
		("Target Speed", f"{target_speed:.1f} km/h", COLOR_YELLOW_GREEN),
		("Command", str(command), COLOR_PINK_RED),
	]

	segments = []
	total_w = 0
	for label, value, _accent in items:
		label_w = font_bold.getbbox(label)[2] - font_bold.getbbox(label)[0]
		value_w = font_bold.getbbox(value)[2] - font_bold.getbbox(value)[0]
		seg_w = label_w + 12 + value_w
		segments.append((label, value, seg_w, label_w))
		total_w += seg_w

	gap = max(34, int((x2 - x1) * 0.04))
	total_w += gap * (len(segments) - 1)
	x_pos = max(x1 + pad, int((x1 + x2 - total_w) // 2))

	for idx, ((label, value, seg_w, label_w), (_l2, _v2, accent)) in enumerate(zip(segments, items)):
		draw_text_pil(frame, label, x_pos, y_text, COLOR_WHITE, font_bold, bold=True)
		draw_text_pil(frame, value, x_pos + label_w + 12, y_text, accent, font_bold, bold=True)
		x_pos += seg_w
		if idx < len(segments) - 1:
			x_pos += gap

	return hud_h


def draw_prediction_panel(frame, ground_truth_route, predicted_route, top_h, measurement):
	h, w = frame.shape[:2]
	margin = max(8, int(min(h, w) * 0.012))

	panel_x1 = margin
	panel_y1 = top_h + margin
	panel_x2 = panel_x1 + max(150, int(w * 0.175))
	panel_y2 = h - margin
	panel_x2 = min(panel_x2, w - margin)
	panel_y2 = max(panel_y2, panel_y1 + 190)

	draw_rounded_rect(frame, panel_x1, panel_y1, panel_x2, panel_y2, COLOR_DARK_GRAY, radius=7, alpha=0.85)
	cv2.rectangle(frame, (int(panel_x1), int(panel_y1)), (int(panel_x2), int(panel_y2)), COLOR_BRIGHT_GREEN, 1, cv2.LINE_AA)
	font_bold = load_poppins_font(bold=True, size=SIDE_PANEL_FONT_SIZE)
	draw_text_pil(frame, "Trajectory Prediction", panel_x1 + 10, panel_y1 + 12, COLOR_YELLOW_GREEN, font_bold, bold=True)

	target_point = measurement.get("target_point", [0.0, 0.0]) if measurement is not None else [0.0, 0.0]
	if len(target_point) < 2:
		target_point = [0.0, 0.0]
	target_label = "Target"
	target_value = f"({float(target_point[0]):+05.1f}, {float(target_point[1]):+05.1f}) m"

	target_chip_y1 = panel_y1 + 30
	target_chip_y2 = target_chip_y1 + 22
	target_chip_w = max(130, int((panel_x2 - panel_x1) * 0.92))
	draw_rounded_rect(frame, panel_x1 + 8, target_chip_y1, panel_x1 + 8 + target_chip_w, target_chip_y2, COLOR_VERY_DARK, radius=3, alpha=0.90)
	cv2.rectangle(frame, (int(panel_x1 + 8), int(target_chip_y1)), (int(panel_x1 + 8 + target_chip_w), int(target_chip_y2)), COLOR_YELLOW_GREEN, 1, cv2.LINE_AA)
	font_bold = load_poppins_font(bold=True, size=SIDE_PANEL_FONT_SIZE)
	label_x = panel_x1 + 14
	label_w = font_bold.getbbox(target_label)[2] - font_bold.getbbox(target_label)[0]
	draw_text_pil(frame, target_label, label_x, target_chip_y1 + 4, COLOR_WHITE, font_bold, bold=True)
	draw_text_pil(frame, target_value, label_x + label_w + 6, target_chip_y1 + 4, COLOR_BRIGHT_GREEN, font_bold, bold=True)

	inner_pad = 8
	gx1 = panel_x1 + inner_pad
	gx2 = panel_x2 - inner_pad
	gy1 = target_chip_y2 + 10
	gy2 = panel_y2 - inner_pad
	if gx2 - gx1 < 40 or gy2 - gy1 < 50:
		return

	draw_rounded_rect(frame, gx1, gy1, gx2, gy2, COLOR_VERY_DARK, radius=60, alpha=0.92)
	cv2.rectangle(frame, (int(gx1), int(gy1)), (int(gx2), int(gy2)), COLOR_BRIGHT_GREEN, 1, cv2.LINE_AA)
	gt_color = (255, 133, 133)
	pred_color = (88, 228, 212)

	legend_y = gy1 + 14
	font_bold = load_poppins_font(bold=True, size=SIDE_PANEL_FONT_SIZE-1)
	# Legend colors exactly match route colors (convert BGR to RGB for PIL text).
	gt_color_rgb = (gt_color[2], gt_color[1], gt_color[0])
	pred_color_rgb = (pred_color[2], pred_color[1], pred_color[0])
	cv2.line(frame, (int(gx1 + 6), int(legend_y - 4)), (int(gx1 + 18), int(legend_y - 4)), gt_color, 2, cv2.LINE_AA)
	draw_text_pil(frame, "Ground Truth", gx1 + 22, legend_y - 10, gt_color_rgb, font_bold, bold=True)
	cv2.line(frame, (int(gx1 + 6), int(legend_y + 10)), (int(gx1 + 18), int(legend_y + 10)), pred_color, 2, cv2.LINE_AA)
	draw_text_pil(frame, "Prediction", gx1 + 22, legend_y+4, pred_color_rgb, font_bold, bold=True)

	center_x = (gx1 + gx2) // 2
	bottom_y = gy2 - 10
	grid_top = gy1 + 38

	cv2.line(frame, (center_x, grid_top), (center_x, gy2 - 4), (49, 83, 119), 1, cv2.LINE_AA)
	cv2.line(frame, (gx1 + 5, bottom_y), (gx2 - 5, bottom_y), (49, 83, 119), 1, cv2.LINE_AA)

	lat_limit = 8.0
	lon_limit = 20.0
	for route in (ground_truth_route, predicted_route):
		if route is None or len(route) == 0:
			continue
		lat_limit = max(lat_limit, float(np.max(np.abs(route[:, 1]))) + 1.0)
		lon_limit = max(lon_limit, float(np.max(route[:, 0])) + 2.0)

	sx = (gx2 - gx1 - 20) / (2.0 * lat_limit)
	sy = (gy2 - gy1 - 26) / max(5.0, lon_limit + 3.0)
	scale = max(1e-3, min(sx, sy))

	def to_points(route):
		if route is None or len(route) == 0:
			return None
		pts = []
		for longitudinal, lateral in route:
			px = int(center_x + float(lateral) * scale)
			py = int(bottom_y - float(longitudinal) * scale)
			px = max(gx1 + 3, min(gx2 - 3, px))
			py = max(grid_top + 3, min(gy2 - 3, py))
			pts.append((px, py))
		return np.array(pts, dtype=np.int32)

	gt_pts = to_points(ground_truth_route)
	pred_pts = to_points(predicted_route)

	if gt_pts is not None and len(gt_pts) >= 2:
		cv2.polylines(frame, [gt_pts], False, gt_color, 2, cv2.LINE_AA)
		for p in gt_pts[:: max(1, len(gt_pts) // 10)]:
			cv2.circle(frame, tuple(p), 2, gt_color, -1, cv2.LINE_AA)

	if pred_pts is not None and len(pred_pts) >= 2:
		cv2.polylines(frame, [pred_pts], False, pred_color, 2, cv2.LINE_AA)
		for p in pred_pts[:: max(1, len(pred_pts) // 10)]:
			cv2.circle(frame, tuple(p), 2, pred_color, -1, cv2.LINE_AA)

	ego_w = max(10, int((0.9 * scale)))
	ego_h = max(14, int((1.8 * scale)))
	cv2.rectangle(
		frame,
		(center_x - ego_w // 2, bottom_y - ego_h),
		(center_x + ego_w // 2, bottom_y),
		(250, 215, 95),
		-1,
		cv2.LINE_AA,
	)
	cv2.rectangle(
		frame,
		(center_x - ego_w // 2, bottom_y - ego_h),
		(center_x + ego_w // 2, bottom_y),
		(255, 255, 255),
		1,
		cv2.LINE_AA,
	)


def render_frame(image, ground_truth_route, predicted_route, measurement, results, frame_idx, logo_rgba=None):
	frame = image.copy()
	h, w = frame.shape[:2]
	margin = max(8, int(min(h, w) * 0.01))

	top_h = draw_top_hud(frame, measurement, results, frame_idx)
	draw_prediction_panel(frame, ground_truth_route, predicted_route, top_h, measurement)

	if logo_rgba is not None:
		logo_x = w - logo_rgba.shape[1] - margin
		logo_y = h - logo_rgba.shape[0] - margin
		draw_filled_rect(
			frame,
			logo_x - 6,
			logo_y - 6,
			logo_x + logo_rgba.shape[1] + 6,
			logo_y + logo_rgba.shape[0] + 6,
			(6, 16, 30),
			alpha=0.46,
		)
		blend_icon_rgba(frame, logo_rgba, logo_x, logo_y)

	return frame


def iter_frame_ids(scene_path):
	rgb_dir = scene_path / "rgb"
	for image_path in sorted(rgb_dir.glob("*.jpg")):
		yield image_path.stem


def generate_video(scene_path, output_path, config_path, student_ckpt, fps, device=None):
	config, net = load_config_and_model(config_path, student_ckpt, device)
	model_device = get_model_device(net)
	preprocessor = InputPreprocessor(config, device=model_device)

	sample_h = int(config.camera_height) if getattr(config, "camera_height", 0) else 720
	logo_rgba = load_logo_rgba(DEFAULT_LOGO_PATH, target_height=max(36, int(sample_h * 0.10)))

	video_writer = None
	frame_ids = list(iter_frame_ids(scene_path))
	if not frame_ids:
		raise FileNotFoundError(f"No RGB frames found in {scene_path / 'rgb'}")

	for frame_id in tqdm(frame_ids, desc="Generating inference video"):
		image_path = scene_path / "rgb" / f"{frame_id}.jpg"
		lidar_path = resolve_lidar_path(scene_path, frame_id)
		measurement_path = scene_path / "measurements" / f"{frame_id}.json.gz"

		if not (image_path.exists() and lidar_path is not None and measurement_path.exists()):
			continue

		image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
		lidar_points = load_lidar_points(lidar_path)
		measurement = load_measurement(measurement_path)
		measurement["speed"] = 0.0  # Override speed for better visualization of predictions at low speeds.
		target_point = measurement["target_point"]
		speed = float(measurement["speed"])
		command = int(measurement["command"])

		rgb, lidar_bev, target_point_tensor, velocity_tensor, command_tensor = preprocessor.prepare_inputs(
			image,
			lidar_points,
			target_point,
			speed,
			command,
		)

		with torch.no_grad():
			output = net(rgb, lidar_bev, target_point_tensor, velocity_tensor, command_tensor)

		results = postprocess_model_outputs(output, config, net, ego_velocity=velocity_tensor.item())

		predicted_route = results["checkpoints"]
		if predicted_route is not None:
			predicted_route = normalize_route(predicted_route)

		ground_truth_route = normalize_route(measurement["route"], target_length=len(predicted_route) if predicted_route is not None else None)
		frame_rgb = render_frame(
			image=image,
			ground_truth_route=ground_truth_route,
			predicted_route=predicted_route,
			measurement=measurement,
			results=results,
			frame_idx=int(frame_id),
			logo_rgba=logo_rgba,
		)

		if video_writer is None:
			height, width = frame_rgb.shape[:2]
			video_writer = cv2.VideoWriter(
				str(output_path),
				cv2.VideoWriter_fourcc(*"mp4v"),
				fps,
				(width, height),
			)

		video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

	if video_writer is not None:
		video_writer.release()
		print(f"Video saved at: {output_path}")


def parse_args():
	parser = argparse.ArgumentParser(description="Generate a route-comparison inference video for Transfuser++.")
	parser.add_argument("--scene-path", type=Path, default=DEFAULT_SCENE_PATH)
	parser.add_argument("--output", type=Path, default=DEFAULT_VIDEO_PATH)
	parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
	parser.add_argument("--student-ckpt", type=Path, default=DEFAULT_STUDENT_CKPT)
	parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
	parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to run on: auto, cpu, cuda, cuda:0, mps")
	return parser.parse_args()


def main():
	args = parse_args()
	generate_video(
		scene_path=args.scene_path.resolve(),
		output_path=args.output.resolve(),
		config_path=args.config_path.resolve(),
		student_ckpt=args.student_ckpt.resolve(),
		fps=args.fps,
		device=args.device,
	)


if __name__ == "__main__":
	main()
