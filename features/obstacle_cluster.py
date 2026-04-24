import importlib
from typing import Dict, List, Tuple

import numpy as np

from utils.bbox_pick import fit_obb_xy


class ObstacleCluster:
    """点云聚类与包围框提取流程（ROI -> 下采样 -> DBSCAN -> 框）。"""

    def points_in_roi(self, points_xyz: np.ndarray, roi_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        if points_xyz is None or len(points_xyz) == 0:
            return np.empty((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool)
        if not roi_config.get("use_roi", True):
            pts = np.asarray(points_xyz[:, :3], dtype=np.float64)
            return pts, np.ones((len(pts),), dtype=bool)
        x_min, x_max = sorted((float(roi_config["roi_x_min"]), float(roi_config["roi_x_max"])))
        y_min, y_max = sorted((float(roi_config["roi_y_min"]), float(roi_config["roi_y_max"])))
        z_min, z_max = sorted((float(roi_config["roi_z_min"]), float(roi_config["roi_z_max"])))
        pts = np.asarray(points_xyz[:, :3], dtype=np.float64)
        # 与 C++ pointsInRoi 对齐：使用严格不等号
        mask = (
            (pts[:, 0] > x_min) & (pts[:, 0] < x_max) &
            (pts[:, 1] > y_min) & (pts[:, 1] < y_max) &
            (pts[:, 2] > z_min) & (pts[:, 2] < z_max)
        )
        return pts[mask], mask

    def pcd_down_sample(self, points_xyz: np.ndarray, voxel_size: float, roi_config: Dict) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        if points_xyz is None or len(points_xyz) == 0:
            return np.empty((0, 3), dtype=np.float64), {}
        if voxel_size <= 0:
            voxel_map = {i: [i] for i in range(len(points_xyz))}
            return np.asarray(points_xyz, dtype=np.float64), voxel_map

        pts = np.asarray(points_xyz, dtype=np.float64)
        x_min = float(roi_config.get("roi_x_min", np.min(pts[:, 0])))
        y_min = float(roi_config.get("roi_y_min", np.min(pts[:, 1])))
        z_min = float(roi_config.get("roi_z_min", np.min(pts[:, 2])))
        # 先平移到 ROI 局部坐标再体素化，避免负坐标与比例不匹配导致 key 冲突
        shifted = np.empty_like(pts)
        shifted[:, 0] = pts[:, 0] - x_min
        shifted[:, 1] = pts[:, 1] - y_min
        shifted[:, 2] = pts[:, 2] - z_min
        voxel_keys = np.floor(shifted / float(voxel_size)).astype(np.int64)
        key_to_ds_idx = {}
        down_points: List[np.ndarray] = []
        voxel_indices_map: Dict[int, List[int]] = {}
        for idx, key in enumerate(voxel_keys):
            # 使用三元组作为体素 key，彻底避免不同体素映射冲突
            voxel_key = (int(key[0]), int(key[1]), int(key[2]))
            ds_idx = key_to_ds_idx.get(voxel_key)
            if ds_idx is None:
                ds_idx = len(down_points)
                key_to_ds_idx[voxel_key] = ds_idx
                down_points.append(pts[idx].copy())
                voxel_indices_map[ds_idx] = [idx]
            else:
                voxel_indices_map[ds_idx].append(idx)
        down_points_np = np.array(down_points, dtype=np.float64)
        return down_points_np, voxel_indices_map

    def pcd_obstacle_cluster(self, points_xyz: np.ndarray, eps: float, min_points: int) -> np.ndarray:
        sklearn_cluster = importlib.import_module("sklearn.cluster")
        DBSCAN = getattr(sklearn_cluster, "DBSCAN")
        if points_xyz is None or len(points_xyz) == 0:
            return np.array([], dtype=np.int32)
        labels = DBSCAN(eps=float(eps), min_samples=int(min_points)).fit_predict(points_xyz)
        return np.asarray(labels, dtype=np.int32)

    def obtained_cluster_box(self, points_xyz: np.ndarray) -> Dict:
        if points_xyz is None or len(points_xyz) < 3:
            return {}
        pts = np.asarray(points_xyz, dtype=np.float64)
        z_min = float(np.min(pts[:, 2]))
        z_max = float(np.max(pts[:, 2]))
        z_c = 0.5 * (z_min + z_max)
        h = max(z_max - z_min, 0.1)

        xy = pts[:, :2].astype(np.float32)
        box = self._fit_xy_box(xy)
        if box is None:
            return {}
        x_c, y_c, l, w, yaw = box
        return {
            "x": float(x_c),
            "y": float(y_c),
            "z": float(z_c),
            "l": float(max(l, 0.1)),
            "w": float(max(w, 0.1)),
            "h": float(h),
            "yaw": float(yaw),
        }

    def obtained_cluster_box_lshape(self, points_xyz: np.ndarray) -> Dict:
        if points_xyz is None or len(points_xyz) < 3:
            return {}
        pts = np.asarray(points_xyz, dtype=np.float64)
        min_z = float(np.min(pts[:, 2]))
        max_z = float(np.max(pts[:, 2]))
        epsilon = 0.01
        h = max(max_z - min_z, epsilon)
        z_c = min_z + 0.5 * h

        xy = pts[:, :2]
        theta_star = self._optimize_lshape_theta(xy, 0.0, 0.5 * np.pi)
        c = np.cos(theta_star)
        s = np.sin(theta_star)
        c1 = xy[:, 0] * c + xy[:, 1] * s
        c2 = -xy[:, 0] * s + xy[:, 1] * c
        min_c1, max_c1 = float(np.min(c1)), float(np.max(c1))
        min_c2, max_c2 = float(np.min(c2)), float(np.max(c2))

        # 与 C++ CreateLShapeFittingBox 等价：中心在两组对角点中点
        p1 = np.array([c * min_c1 - s * min_c2, s * min_c1 + c * min_c2], dtype=np.float64)
        p2 = np.array([c * max_c1 - s * max_c2, s * max_c1 + c * max_c2], dtype=np.float64)
        center = 0.5 * (p1 + p2)
        l = max(max_c1 - min_c1, epsilon)
        w = max(max_c2 - min_c2, epsilon)

        return {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(z_c),
            "l": float(l),
            "w": float(w),
            "h": float(h),
            "yaw": float(theta_star),
        }

    def _calc_closeness_criterion(self, c1: np.ndarray, c2: np.ndarray) -> float:
        min_c1, max_c1 = float(np.min(c1)), float(np.max(c1))
        min_c2, max_c2 = float(np.min(c2)), float(np.max(c2))
        d1 = np.minimum(max_c1 - c1, c1 - min_c1) ** 2
        d2 = np.minimum(max_c2 - c2, c2 - min_c2) ** 2
        d_min = 0.1 ** 2
        d_max = 0.4 ** 2
        d = np.minimum(d1, d2)
        valid = d <= d_max
        if not np.any(valid):
            return 0.0
        d_clip = np.maximum(d[valid], d_min)
        return float(np.sum(1.0 / d_clip))

    def _optimize_lshape_theta(self, points_xy: np.ndarray, min_angle: float, max_angle: float) -> float:
        best_theta = 0.0
        best_q = -1.0
        angle_step = np.deg2rad(2.0)  # 对齐 C++ angle_resolution
        theta = float(min_angle)
        while theta <= max_angle + 0.01:
            c = np.cos(theta)
            s = np.sin(theta)
            c1 = points_xy[:, 0] * c + points_xy[:, 1] * s
            c2 = -points_xy[:, 0] * s + points_xy[:, 1] * c
            q = self._calc_closeness_criterion(c1, c2)
            if q > best_q:
                best_q = q
                best_theta = theta
            theta += angle_step
        return float(best_theta)

    def _fit_xy_box(self, points_xy: np.ndarray):
        try:
            cv2 = importlib.import_module("cv2")
            # 与 C++ obtainedClusterBox 对齐：先凸包，再最小外接矩形
            hull = cv2.convexHull(points_xy.reshape(-1, 1, 2))
            rect = cv2.minAreaRect(hull)
            (cx, cy), (size_w, size_h), angle_deg = rect
            l = float(size_w)
            w = float(size_h)
            yaw = np.deg2rad(float(angle_deg))
            if l < w:
                l, w = w, l
                yaw += np.pi / 2.0
            return float(cx), float(cy), float(l), float(w), float(yaw)
        except Exception:
            fit = fit_obb_xy(points_xy)
            if fit is None:
                return None
            return fit

    def cluster(self, points_xyz: np.ndarray, config: Dict) -> Tuple[List[Dict], np.ndarray]:
        roi_points, roi_mask = self.points_in_roi(points_xyz, config)
        if len(roi_points) == 0:
            return [], roi_mask

        voxel_size = float(config.get("voxel_size", 0.0))
        ds_points, voxel_map = self.pcd_down_sample(roi_points, voxel_size, config)
        labels = self.pcd_obstacle_cluster(ds_points, config["eps"], config["min_points"])
        if len(labels) == 0:
            return [], roi_mask

        boxes: List[Dict] = []
        min_points = int(config["min_points"])
        max_cluster_points = int(config.get("max_cluster_points", 2_000_000))
        use_lshape = bool(config.get("use_lshape", False))
        unique_labels = [lab for lab in np.unique(labels) if lab != -1]
        for lab in unique_labels:
            ds_cluster_idx = np.where(labels == lab)[0]
            roi_indices: List[int] = []
            for ds_idx in ds_cluster_idx:
                roi_indices.extend(voxel_map.get(int(ds_idx), []))
            if len(roi_indices) < min_points or len(roi_indices) > max_cluster_points:
                continue
            cluster_pts = roi_points[np.asarray(roi_indices, dtype=np.int64)]
            box = self.obtained_cluster_box_lshape(cluster_pts) if use_lshape else self.obtained_cluster_box(cluster_pts)
            if box and self._box_passes_size_filter(box, config):
                boxes.append(box)
        return boxes, roi_mask

    def _box_passes_size_filter(self, box: Dict, config: Dict) -> bool:
        if not bool(config.get("use_size_filter", False)):
            return True
        l_min, l_max = sorted((float(config.get("l_min", 0.0)), float(config.get("l_max", 1e9))))
        w_min, w_max = sorted((float(config.get("w_min", 0.0)), float(config.get("w_max", 1e9))))
        h_min, h_max = sorted((float(config.get("h_min", 0.0)), float(config.get("h_max", 1e9))))
        l = float(box.get("l", 0.0))
        w = float(box.get("w", 0.0))
        h = float(box.get("h", 0.0))
        return (l_min <= l <= l_max) and (w_min <= w <= w_max) and (h_min <= h <= h_max)
