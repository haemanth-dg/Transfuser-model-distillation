"""
KD-only dataset loader for student distillation training.

Expected sample keys:
    rgb, lidar, target_point, command, target_speed, speed, route,
    center_heatmap_target, wh_target, yaw_class_target, yaw_res_target,
    offset_target, velocity_target, brake_target, pixel_weight, avg_factor
"""

import gzip
import json
import os
import random

import cv2
import laspy
import numpy as np
from torch.utils.data import Dataset
from imgaug import augmenters as ia

from . import transfuser_utils as t_u
from . import gaussian_target as g_t
from .center_net import angle2class


class Custom_Data(Dataset):  # pylint: disable=locally-disabled, invalid-name
    """Dataset that loads only the fields needed by kd_trainer.py."""

    def __init__(
        self,
        root,
        config,
        estimate_class_distributions=False,
        estimate_sem_distribution=False,
        shared_dict=None,
        rank=0,
        validation=False,
    ):
        del estimate_class_distributions
        del estimate_sem_distribution
        self.config = config
        self.validation = validation
        self.data_cache = shared_dict
        self.samples = []
        self.image_augmenter_func = image_augmenter(getattr(config, "color_aug_prob", 0.0), cutout=getattr(config, "use_cutout", False))
        self.lidar_augmenter_func = lidar_augmenter(getattr(config, "lidar_aug_prob", 0.0), cutout=getattr(config, "use_cutout", False))

        if not isinstance(root, (list, tuple)):
            root = [root]

        for sub_root in root:
            if not os.path.isdir(sub_root):
                continue

            for dirpath, dirnames, _filenames in os.walk(sub_root):
                del dirnames
                measurement_dir = os.path.join(dirpath, "measurements")
                rgb_dir = os.path.join(dirpath, "rgb")
                lidar_dir = os.path.join(dirpath, "lidar")
                boxes_dir = os.path.join(dirpath, "boxes")

                if not (os.path.isdir(measurement_dir) and os.path.isdir(rgb_dir) and os.path.isdir(lidar_dir)):
                    continue
                if not os.path.isdir(boxes_dir):
                    raise FileNotFoundError(f"Missing boxes folder for route: {boxes_dir}")

                route_name = os.path.basename(dirpath)
                is_validation_route = "validation" in route_name.lower()
                if self.validation and not is_validation_route:
                    continue
                if not self.validation and is_validation_route:
                    continue

                for file_name in sorted(os.listdir(measurement_dir)):
                    if not file_name.endswith(".json.gz"):
                        continue
                    frame_id = file_name[:-8]
                    if int(frame_id) % int(getattr(self.config, "train_sampling_rate", 1)) != 0:
                        continue
                    measurement_path = os.path.join(measurement_dir, file_name)
                    rgb_path = os.path.join(rgb_dir, f"{frame_id}.jpg")
                    lidar_path = os.path.join(lidar_dir, f"{frame_id}.npy")
                    lidar_laz_path = os.path.join(lidar_dir, f"{frame_id}.laz")
                    boxes_path = os.path.join(boxes_dir, f"{frame_id}.json.gz")
                    if not (
                        os.path.isfile(measurement_path)
                        and os.path.isfile(rgb_path)
                        and (os.path.isfile(lidar_path) or os.path.isfile(lidar_laz_path))
                        and os.path.isfile(boxes_path)
                    ):
                        continue
                    self.samples.append((dirpath, frame_id))

        if rank == 0:
            print(f"Loaded {len(self.samples)} KD samples from {len(root)} root folder(s)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        route_dir, frame_id = self.samples[index]

        measurement = self._read_measurement(os.path.join(route_dir, "measurements", f"{frame_id}.json.gz"))
        boxes = self._read_boxes(os.path.join(route_dir, "boxes", f"{frame_id}.json.gz"))
        rgb = self._load_rgb(os.path.join(route_dir, "rgb", f"{frame_id}.jpg"))
        rgb_augmented_path = os.path.join(route_dir, "rgb_augmented", f"{frame_id}.jpg")
        rgb_augmented = self._load_rgb(rgb_augmented_path) if os.path.exists(rgb_augmented_path) else rgb
        lidar_npy_path = os.path.join(route_dir, "lidar", f"{frame_id}.npy")
        lidar_laz_path = os.path.join(route_dir, "lidar", f"{frame_id}.laz")
        lidar = self._load_lidar(lidar_npy_path, lidar_laz_path)

        # print(f"Augment: {self.config.augment}")
        if getattr(self.config, "augment", False) and random.random() <= float(getattr(self.config, "augment_percentage", 0.0)):
            augment_sample = True
            aug_rotation = float(measurement.get("augmentation_rotation", 0.0))
            aug_translation = float(measurement.get("augmentation_translation", 0.0))
        else:
            augment_sample = False
            aug_rotation = 0.0
            aug_translation = 0.0

        if getattr(self.config, "augment", False) and augment_sample:
            if getattr(self.config, "use_color_aug", False):
                processed_image = self.image_augmenter_func(image=rgb_augmented)
            else:
                processed_image = rgb_augmented
        else:
            if getattr(self.config, "use_color_aug", False):
                processed_image = self.image_augmenter_func(image=rgb)
            else:
                processed_image = rgb

        if getattr(self.config, "augment", False):
            lidar = self.align(
                lidar,
                measurement,
                measurement,
                y_augmentation=aug_translation,
                yaw_augmentation=aug_rotation,
            )

        lidar_bev = self.lidar_to_histogram_features(
            lidar,
            use_ground_plane=bool(getattr(self.config, "use_ground_plane", False)),
        )
        if getattr(self.config, "augment", False):
            lidar_bev = self.lidar_augmenter_func(image=np.transpose(lidar_bev, (1, 2, 0)))
            lidar_bev = np.transpose(lidar_bev, (2, 0, 1))

        route = np.asarray(measurement.get("route", []), dtype=np.float32)
        route = self._normalize_route(route)
        route = self.augment_route(route, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        if getattr(self.config, "smooth_route", True):
            route = self.smooth_path(route)

        target_point = np.asarray(measurement.get("target_point", [0.0, 0.0]), dtype=np.float32)
        target_point = self.augment_target_point(
            target_point,
            y_augmentation=aug_translation,
            yaw_augmentation=aug_rotation,
        )
        command = t_u.command_to_one_hot(int(measurement.get("command", 0))).astype(np.float32)
        target_speed = np.int64(self.get_indices_speed_angle(
            target_speed=float(measurement.get("target_speed", 0.0)),
            brake=False,
            angle=float(measurement.get("angle", 0.0)),
        )[0])

        bboxes, _ = self.parse_bounding_boxes(
            boxes,
            future_boxes=None,
            y_augmentation=aug_translation,
            yaw_augmentation=aug_rotation,
        )
        if len(bboxes) > 0:
            bboxes = np.asarray(bboxes, dtype=np.float32)
        else:
            bboxes = np.zeros((0, 8), dtype=np.float32)

        feat_h = int(self.config.lidar_resolution_height // self.config.bev_down_sample_factor)
        feat_w = int(self.config.lidar_resolution_width // self.config.bev_down_sample_factor)
        bbox_targets, avg_factor = self.get_targets(bboxes, feat_h, feat_w)

        return {
            "rgb": np.transpose(processed_image, (2, 0, 1)),
            "lidar": lidar_bev,
            "target_point": target_point,
            "command": command,
            "target_speed": target_speed,
            "speed": np.float32(0.0),
            "route": route,
            "center_heatmap_target": bbox_targets["center_heatmap_target"],
            "wh_target": bbox_targets["wh_target"],
            "yaw_class_target": bbox_targets["yaw_class_target"],
            "yaw_res_target": bbox_targets["yaw_res_target"],
            "offset_target": bbox_targets["offset_target"],
            "velocity_target": bbox_targets["velocity_target"],
            "brake_target": bbox_targets["brake_target"],
            "pixel_weight": bbox_targets["pixel_weight"],
            "avg_factor": np.float32(avg_factor),
        }

    def _read_measurement(self, path):
        return self._read_json_gz(path)

    def _read_boxes(self, path):
        boxes = self._read_json_gz(path)
        if isinstance(boxes, dict):
            if "boxes" in boxes:
                boxes = boxes["boxes"]
            else:
                raise KeyError(f"Malformed boxes payload at {path}: expected list or dict with 'boxes'")
        if not isinstance(boxes, list):
            raise TypeError(f"Malformed boxes payload at {path}: expected list, got {type(boxes).__name__}")
        return boxes

    def _read_json_gz(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        if self.data_cache is not None and path in self.data_cache:
            return self.data_cache[path]

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)

        if self.data_cache is not None:
            self.data_cache[path] = payload
        return payload

    def _load_rgb(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = t_u.crop_array(self.config, image)
        return image.astype(np.uint8)

    def _load_lidar(self, npy_path, laz_path=None):
        if os.path.isfile(npy_path):
            lidar = np.load(npy_path)
        elif laz_path is not None and os.path.isfile(laz_path):
            lidar_las = laspy.read(laz_path)
            lidar = np.stack((lidar_las.x, lidar_las.y, lidar_las.z), axis=1)
        else:
            raise FileNotFoundError(npy_path)
        if lidar.ndim != 2 or lidar.shape[1] < 3:
            raise ValueError(f"Unexpected LiDAR shape: {lidar.shape}")
        return lidar[:, :3].astype(np.float32)

    def _normalize_route(self, route):
        num_points = int(getattr(self.config, "num_route_points", 10))
        if route.size == 0:
            return np.zeros((num_points, 2), dtype=np.float32)
        if route.ndim == 1:
            route = route.reshape(-1, 2)
        if route.shape[0] >= num_points:
            return route[:num_points].astype(np.float32)

        pad_count = num_points - route.shape[0]
        pad_value = np.repeat(route[-1:, :], pad_count, axis=0)
        return np.concatenate([route, pad_value], axis=0).astype(np.float32)

    def augment_route(self, route, y_augmentation=0.0, yaw_augmentation=0.0):
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([
            [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
            [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
        ])
        translation = np.array([[0.0, y_augmentation]])
        return (rotation_matrix.T @ (route - translation).T).T

    def augment_target_point(self, target_point, y_augmentation=0.0, yaw_augmentation=0.0):
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([
            [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
            [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
        ])
        translation = np.array([[0.0], [y_augmentation]])
        pos = np.expand_dims(target_point, axis=1)
        target_point_aug = rotation_matrix.T @ (pos - translation)
        return np.squeeze(target_point_aug)

    def align(self, lidar_0, measurements_0, measurements_1, y_augmentation=0.0, yaw_augmentation=0.0):
        pos_1 = np.array([measurements_1.get("pos_global", [0.0, 0.0])[0], measurements_1.get("pos_global", [0.0, 0.0])[1], 0.0])
        pos_0 = np.array([measurements_0.get("pos_global", [0.0, 0.0])[0], measurements_0.get("pos_global", [0.0, 0.0])[1], 0.0])
        pos_diff = pos_1 - pos_0
        theta_1 = float(measurements_1.get("theta", 0.0))
        theta_0 = float(measurements_0.get("theta", 0.0))
        rot_diff = t_u.normalize_angle(theta_1 - theta_0)

        rotation_matrix = np.array([
            [np.cos(theta_1), -np.sin(theta_1), 0.0],
            [np.sin(theta_1), np.cos(theta_1), 0.0],
            [0.0, 0.0, 1.0],
        ])
        pos_diff = rotation_matrix.T @ pos_diff

        lidar_1 = t_u.algin_lidar(lidar_0, pos_diff, rot_diff)

        pos_diff_aug = np.array([0.0, y_augmentation, 0.0])
        rot_diff_aug = np.deg2rad(yaw_augmentation)
        return t_u.algin_lidar(lidar_1, pos_diff_aug, rot_diff_aug)

    def smooth_path(self, route):
        _, indices = np.unique(route, return_index=True, axis=0)
        route = route[np.sort(indices)]
        return self.iterative_line_interpolation(route)

    def iterative_line_interpolation(self, route):
        interpolated_route_points = []
        min_distance = self.config.dense_route_planner_min_distance
        target_first_distance = 2.5
        last_interpolated_point = np.array([0.0, 0.0])
        current_route_index = 0
        current_point = route[current_route_index]
        last_point = np.array([0.0, 0.0])
        first_iteration = True

        while len(interpolated_route_points) < self.config.num_route_points:
            if not first_iteration:
                current_route_index += 1
                last_point = current_point

            if current_route_index < route.shape[0]:
                current_point = route[current_route_index]
                intersection = t_u.circle_line_segment_intersection(
                    circle_center=last_interpolated_point,
                    circle_radius=min_distance if not first_iteration else target_first_distance,
                    pt1=last_interpolated_point,
                    pt2=current_point,
                    full_line=True,
                )
            else:
                current_point = route[-1]
                last_point = route[-2]
                intersection = t_u.circle_line_segment_intersection(
                    circle_center=last_interpolated_point,
                    circle_radius=min_distance,
                    pt1=last_point,
                    pt2=current_point,
                    full_line=True,
                )

            if len(intersection) > 1:
                point_1 = np.array(intersection[0])
                point_2 = np.array(intersection[1])
                direction = current_point - last_point
                dot_p1_to_last = np.dot(point_1, direction)
                dot_p2_to_last = np.dot(point_2, direction)
                intersection_point = point_1 if dot_p1_to_last > dot_p2_to_last else point_2
                add_point = True
            elif len(intersection) == 1:
                intersection_point = np.array(intersection[0])
                add_point = True
            else:
                add_point = False
                for radius_multiplier in [1.5, 2.0, 3.0]:
                    intersection = t_u.circle_line_segment_intersection(
                        circle_center=last_interpolated_point,
                        circle_radius=min_distance * radius_multiplier,
                        pt1=last_point if current_route_index >= route.shape[0] else last_interpolated_point,
                        pt2=current_point,
                        full_line=True,
                    )
                    if len(intersection) > 0:
                        intersection_point = np.array(intersection[0] if len(intersection) == 1 else intersection[0])
                        add_point = True
                        break
                if not add_point:
                    intersection_point = current_point
                    add_point = True

            if add_point:
                last_interpolated_point = intersection_point
                interpolated_route_points.append(intersection_point)
                min_distance = 1.0

            first_iteration = False

        return np.array(interpolated_route_points)

    def get_indices_speed_angle(self, target_speed, brake, angle):
        target_speed_bins = np.array(self.config.target_speed_bins)
        angle_bins = np.array(self.config.angle_bins)
        target_speed_index = np.digitize(x=target_speed, bins=target_speed_bins)
        if brake:
            target_speed_index = 0
        else:
            target_speed_index += 1
        angle_index = np.digitize(x=angle, bins=angle_bins)
        return target_speed_index, angle_index

    def get_targets(self, gt_bboxes, feat_h, feat_w):
        img_h = self.config.lidar_resolution_height
        img_w = self.config.lidar_resolution_width

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = np.zeros([self.config.num_bb_classes, feat_h, feat_w], dtype=np.float32)
        wh_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
        offset_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
        yaw_class_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
        yaw_res_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
        velocity_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
        brake_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
        pixel_weight = np.zeros([2, feat_h, feat_w], dtype=np.float32)

        if not gt_bboxes.shape[0] > 0:
            target_result = {
                "center_heatmap_target": center_heatmap_target,
                "wh_target": wh_target,
                "yaw_class_target": yaw_class_target.squeeze(0),
                "yaw_res_target": yaw_res_target,
                "offset_target": offset_target,
                "velocity_target": velocity_target,
                "brake_target": brake_target.squeeze(0),
                "pixel_weight": pixel_weight,
            }
            return target_result, 1

        center_x = gt_bboxes[:, [0]] * width_ratio
        center_y = gt_bboxes[:, [1]] * height_ratio
        gt_centers = np.concatenate((center_x, center_y), axis=1)

        for j, ct in enumerate(gt_centers):
            ctx_int, cty_int = ct.astype(int)
            ctx, cty = ct
            extent_x = gt_bboxes[j, 2] * width_ratio
            extent_y = gt_bboxes[j, 3] * height_ratio

            radius = g_t.gaussian_radius([extent_y, extent_x], min_overlap=0.1)
            radius = max(2, int(radius))
            ind = gt_bboxes[j, -1].astype(int)

            g_t.gen_gaussian_target(center_heatmap_target[ind], [ctx_int, cty_int], radius)

            wh_target[0, cty_int, ctx_int] = extent_x
            wh_target[1, cty_int, ctx_int] = extent_y

            yaw_class, yaw_res = angle2class(gt_bboxes[j, 4], self.config.num_dir_bins)
            yaw_class_target[0, cty_int, ctx_int] = yaw_class
            yaw_res_target[0, cty_int, ctx_int] = yaw_res

            velocity_target[0, cty_int, ctx_int] = gt_bboxes[j, 5]
            brake_target[0, cty_int, ctx_int] = int(round(gt_bboxes[j, 6]))

            offset_target[0, cty_int, ctx_int] = ctx - ctx_int
            offset_target[1, cty_int, ctx_int] = cty - cty_int
            pixel_weight[:, cty_int, ctx_int] = 1.0

        avg_factor = max(1, np.equal(center_heatmap_target, 1).sum())
        target_result = {
            "center_heatmap_target": center_heatmap_target,
            "wh_target": wh_target,
            "yaw_class_target": yaw_class_target.squeeze(0),
            "yaw_res_target": yaw_res_target,
            "offset_target": offset_target,
            "velocity_target": velocity_target,
            "brake_target": brake_target.squeeze(0),
            "pixel_weight": pixel_weight,
        }
        return target_result, avg_factor

    def get_bbox_label(self, bbox_dict, y_augmentation=0.0, yaw_augmentation=0.0):
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([
            [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
            [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
        ])

        position = np.array([[bbox_dict["position"][0]], [bbox_dict["position"][1]]])
        translation = np.array([[0.0], [y_augmentation]])

        position_aug = rotation_matrix.T @ (position - translation)
        x, y = position_aug[:2, 0]

        bbox = np.array([x, y, bbox_dict["extent"][0], bbox_dict["extent"][1], 0, 0, 0, 0], dtype=np.float32)
        bbox[4] = t_u.normalize_angle(bbox_dict["yaw"] - aug_yaw_rad)

        if bbox_dict["class"] == "car":
            bbox[5] = bbox_dict["speed"]
            bbox[6] = 0 if np.isnan(bbox_dict["brake"]) else bbox_dict["brake"]
            bbox[7] = 0
        elif bbox_dict["class"] == "walker":
            bbox[5] = bbox_dict["speed"]
            bbox[7] = 1
        elif bbox_dict["class"] == "traffic_light":
            bbox[7] = 2
        elif bbox_dict["class"] == "stop_sign":
            bbox[7] = 3
        return bbox, bbox_dict["position"][2]

    def parse_bounding_boxes(self, boxes, future_boxes=None, y_augmentation=0.0, yaw_augmentation=0.0):
        del future_boxes
        bboxes = []
        future_bboxes = []

        for current_box in boxes:
            if current_box["class"] not in ["traffic_light", "stop_sign", "car", "walker"]:
                continue

            bbox, height = self.get_bbox_label(current_box, y_augmentation, yaw_augmentation)

            if "num_points" in current_box:
                if (
                    current_box["class"] == "walker"
                    and current_box["num_points"] <= self.config.num_lidar_hits_for_detection_walker
                ) or (
                    current_box["class"] == "car"
                    and current_box["num_points"] <= self.config.num_lidar_hits_for_detection_car
                ):
                    continue

            if current_box["class"] == "traffic_light":
                if not current_box["affects_ego"] or current_box["state"] == "Green":
                    continue

            if current_box["class"] == "stop_sign":
                continue

            if (
                bbox[0] <= self.config.min_x
                or bbox[0] >= self.config.max_x
                or bbox[1] <= self.config.min_y
                or bbox[1] >= self.config.max_y
                or height <= self.config.min_z
                or height >= self.config.max_z
            ):
                continue

            bbox = t_u.bb_vehicle_to_image_system(
                bbox, self.config.pixels_per_meter, self.config.min_x, self.config.min_y
            )
            bboxes.append(bbox)

        return bboxes, future_bboxes

    def lidar_to_histogram_features(self, lidar, use_ground_plane):
        lidar = lidar[(lidar[:, 0] >= self.config.min_x) & (lidar[:, 0] < self.config.max_x)]
        lidar = lidar[(lidar[:, 1] >= self.config.min_y) & (lidar[:, 1] < self.config.max_y)]

        if use_ground_plane:
            below = lidar[lidar[:, 2] <= self.config.lidar_split_height]
            above = lidar[lidar[:, 2] > self.config.lidar_split_height]

            below_pixels_x = ((below[:, 0] - self.config.min_x) * self.config.pixels_per_meter).astype(np.int32)
            below_pixels_y = ((below[:, 1] - self.config.min_y) * self.config.pixels_per_meter).astype(np.int32)
            above_pixels_x = ((above[:, 0] - self.config.min_x) * self.config.pixels_per_meter).astype(np.int32)
            above_pixels_y = ((above[:, 1] - self.config.min_y) * self.config.pixels_per_meter).astype(np.int32)

            below_histogram = np.zeros((self.config.lidar_resolution_height, self.config.lidar_resolution_width), dtype=np.float32)
            above_histogram = np.zeros((self.config.lidar_resolution_height, self.config.lidar_resolution_width), dtype=np.float32)

            np.clip(below_pixels_x, 0, self.config.lidar_resolution_width - 1, out=below_pixels_x)
            np.clip(below_pixels_y, 0, self.config.lidar_resolution_height - 1, out=below_pixels_y)
            np.clip(above_pixels_x, 0, self.config.lidar_resolution_width - 1, out=above_pixels_x)
            np.clip(above_pixels_y, 0, self.config.lidar_resolution_height - 1, out=above_pixels_y)

            np.add.at(below_histogram, (below_pixels_x, below_pixels_y), 1)
            np.add.at(above_histogram, (above_pixels_x, above_pixels_y), 1)

            below_histogram = np.clip(below_histogram / self.config.hist_max_per_pixel, 0, 1).T
            above_histogram = np.clip(above_histogram / self.config.hist_max_per_pixel, 0, 1).T
            features = np.stack([below_histogram, above_histogram], axis=0)
        else:
            pixels_x = ((lidar[:, 0] - self.config.min_x) * self.config.pixels_per_meter).astype(np.int32)
            pixels_y = ((lidar[:, 1] - self.config.min_y) * self.config.pixels_per_meter).astype(np.int32)

            histogram = np.zeros((self.config.lidar_resolution_height, self.config.lidar_resolution_width), dtype=np.float32)
            np.clip(pixels_x, 0, self.config.lidar_resolution_width - 1, out=pixels_x)
            np.clip(pixels_y, 0, self.config.lidar_resolution_height - 1, out=pixels_y)
            np.add.at(histogram, (pixels_x, pixels_y), 1)
            features = np.clip(histogram / self.config.hist_max_per_pixel, 0, 1).T[np.newaxis, :, :]

        return features.astype(np.float32)


def image_augmenter(prob=0.2, cutout=False):
    augmentations = [
        ia.Sometimes(prob, ia.GaussianBlur((0, 1.0))),
        ia.Sometimes(prob, ia.AdditiveGaussianNoise(loc=0, scale=(0., 0.05 * 255), per_channel=0.5)),
        ia.Sometimes(prob, ia.Dropout((0.01, 0.1), per_channel=0.5)),
        ia.Sometimes(prob, ia.Multiply((1 / 1.2, 1.2), per_channel=0.5)),
        ia.Sometimes(prob, ia.LinearContrast((1 / 1.2, 1.2), per_channel=0.5)),
        ia.Sometimes(prob, ia.Grayscale((0.0, 0.5))),
        ia.Sometimes(prob, ia.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)),
    ]

    if cutout:
        augmentations.append(ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False)))

    return ia.Sequential(augmentations, random_order=True)


def lidar_augmenter(prob=0.2, cutout=False):
    augmentations = []
    if cutout:
        augmentations.append(ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False, cval=0.0)))
    return ia.Sequential(augmentations, random_order=True)
