# -*- coding: utf-8 -*-
"""多雷达外参标定：按 lidar_id 分雷达施加 xyz+rpy 变换。"""
import json

import numpy as np

from utils.icp_align import (
    auto_voxel_size,
    icp_fitness,
    icp_point_to_point,
    random_downsample,
    voxel_downsample,
)
from utils.move_pcd import RTmatrix2xyzrpy, move_pcd_with_xyzrpy, xyzrpy2RTmatrix

PARAM_KEYS = ("dx", "dy", "dz", "roll", "pitch", "yaw")

LIDAR_COLORS = [
    (1.0, 0.25, 0.25, 0.88),
    (0.25, 0.92, 0.35, 0.88),
    (0.25, 0.55, 1.0, 0.88),
    (1.0, 0.85, 0.15, 0.88),
    (0.92, 0.25, 0.92, 0.88),
    (0.25, 0.9, 0.9, 0.88),
    (1.0, 0.5, 0.1, 0.88),
    (0.65, 0.65, 0.65, 0.88),
]

MAX_DISPLAY_POINTS = 120000


def default_lidar_color(index: int):
    return LIDAR_COLORS[index % len(LIDAR_COLORS)]


def lidar_id_to_str(lid) -> str:
    return str(int(lid)) if float(lid).is_integer() else str(lid)


def unique_lidar_ids(points: np.ndarray) -> list:
    ids = np.unique(points[:, -1])
    return sorted(ids.tolist(), key=lambda x: (isinstance(x, float), x))


def mask_lidar(points: np.ndarray, lid_id) -> np.ndarray:
    return np.isclose(points[:, -1], lid_id, rtol=0, atol=1e-4)


def same_lidar_id(a, b) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), rtol=0, atol=1e-4))
    except (TypeError, ValueError):
        return a == b


def get_offset_for_lidar(lid, offsets: dict):
    for key in (lid, str(lid)):
        if key in offsets:
            return offsets[key]
    if float(lid).is_integer():
        ik = int(lid)
        if ik in offsets:
            return offsets[ik]
        if str(ik) in offsets:
            return offsets[str(ik)]
    for key, val in offsets.items():
        try:
            if np.isclose(float(key), float(lid)):
                return val
        except (TypeError, ValueError):
            continue
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def structured_to_points_array(structured, fields):
    """将 structured 点云拼为 (N, C) 数组，列顺序与 fields 一致（跳过 _）。"""
    valid = [f for f in fields if f != "_"]
    if not valid:
        raise ValueError("点云无有效字段")
    cols = [np.asarray(structured[f], dtype=np.float64).reshape(-1) for f in valid]
    n = cols[0].shape[0]
    for c in cols[1:]:
        if c.shape[0] != n:
            raise ValueError("各字段点数不一致")
    return np.column_stack(cols)


def validate_lidar_id_last_column(fields) -> tuple:
    valid = [f for f in fields if f != "_"]
    if len(valid) < 4:
        return False, "点云至少需要 x、y、z 及最后一列 lidar_id"
    if valid[-1].lower() != "lidar_id":
        return (
            False,
            "最后一列字段须为 lidar_id，当前最后一列为「{}」".format(valid[-1]),
        )
    return True, None


def apply_extrinsic_offsets(points: np.ndarray, offsets: dict, degrees: bool = True) -> np.ndarray:
    if points is None or len(points) == 0:
        return points
    parts = []
    for lid in unique_lidar_ids(points):
        sub = points[mask_lidar(points, lid)].copy()
        xyz_rpy = get_offset_for_lidar(lid, offsets)
        if any(abs(v) > 1e-12 for v in xyz_rpy):
            sub = move_pcd_with_xyzrpy(sub, xyz_rpy, degrees=degrees)
        parts.append(sub)
    return np.concatenate(parts, axis=0)


def apply_offsets_to_structured(structured, fields, offsets: dict, degrees: bool = True):
    """按雷达更新 structured 数组中的各列。"""
    pts = structured_to_points_array(structured, fields)
    corrected = apply_extrinsic_offsets(pts, offsets, degrees=degrees)
    valid = [f for f in fields if f != "_"]
    out = structured.copy()
    for i, name in enumerate(valid):
        out[name] = corrected[:, i].astype(structured[name].dtype, copy=False)
    return out


def resolve_lidar1_id(lidar_ids: list):
    """解析参考雷达 lidar_id=1；不存在则抛出 ValueError。"""
    for lid in lidar_ids:
        try:
            if np.isclose(float(lid), 1.0):
                return lid
        except (TypeError, ValueError):
            pass
        if str(lid).strip() == "1":
            return lid
    raise ValueError(
        "当前点云未找到 lidar_id=1 的雷达，无法作为 ICP 参考。"
        " 现有机型: {}".format(lidar_ids)
    )


def _transform_xyz(points_xyz: np.ndarray, xyz_rpy, degrees: bool = True) -> np.ndarray:
    if points_xyz is None or len(points_xyz) == 0:
        return points_xyz
    padded = np.column_stack(
        [points_xyz[:, :3], np.zeros((len(points_xyz), 1), dtype=np.float64)]
    )
    moved = move_pcd_with_xyzrpy(padded, xyz_rpy, degrees=degrees)
    return moved[:, :3]


def _sort_lidar_ids_numeric(lidar_ids: list, ref_lid) -> list:
    """除参考雷达外，按 id 数值升序（便于链式相邻雷达优先）。"""

    def _num(lid):
        try:
            return float(lid)
        except (TypeError, ValueError):
            return float("inf")

    others = [lid for lid in lidar_ids if not same_lidar_id(lid, ref_lid)]
    return sorted(others, key=_num)


def _prepare_icp_cloud(
    xyz: np.ndarray, voxel_size: float, max_points: int
) -> np.ndarray:
    if xyz is None or len(xyz) == 0:
        return xyz
    out = voxel_downsample(xyz, voxel_size)
    return random_downsample(out, max_points)


def _lidar_xyz_transformed(pts_full, lid, offset) -> np.ndarray:
    raw = pts_full[mask_lidar(pts_full, lid)][:, :3]
    return _transform_xyz(raw, offset, degrees=True)


def _offset_apply_icp_delta(old_off, t_icp) -> list:
    rt_old = xyzrpy2RTmatrix(old_off, degrees=True)
    rt_new = t_icp @ rt_old
    return [float(v) for v in RTmatrix2xyzrpy(rt_new, degrees=True).tolist()]


def _offset_delta_norm(old_off, new_off) -> float:
    return float(np.linalg.norm(np.asarray(new_off, dtype=np.float64) - np.asarray(old_off, dtype=np.float64)))


def _build_merged_target_excluding(
    pts_full,
    offsets,
    lidar_ids,
    exclude_lid,
    voxel_size: float,
    max_points_per_lidar: int,
    max_total_points: int = 120000,
) -> np.ndarray:
    """融合除 exclude_lid 外所有雷达当前位姿点云，作为 ICP 目标。"""
    parts = []
    for lid in lidar_ids:
        if same_lidar_id(lid, exclude_lid):
            continue
        xyz = _lidar_xyz_transformed(pts_full, lid, get_offset_for_lidar(lid, offsets))
        if len(xyz) < 10:
            continue
        parts.append(_prepare_icp_cloud(xyz, voxel_size, max_points_per_lidar))
    if not parts:
        return None
    merged = np.vstack(parts)
    return random_downsample(merged, max_total_points)


def _run_icp_pair(source_xyz, target_xyz, voxel_size, max_pts, max_iter, max_dist):
    src = _prepare_icp_cloud(source_xyz, voxel_size, max_pts)
    tgt = _prepare_icp_cloud(target_xyz, voxel_size, max_pts)
    if src is None or tgt is None or len(src) < 10 or len(tgt) < 10:
        return None, np.inf, False
    return icp_point_to_point(
        src, tgt, max_iterations=max_iter, max_correspondence_distance=max_dist
    )


def icp_multi_refine_align(
    structured,
    fields,
    offsets: dict,
    lidar_ids: list,
    ref_lid=None,
    max_points_per_lidar: int = 50000,
    max_iterations: int = 40,
    max_correspondence_distance: float = 2.0,
    global_refine_rounds: int = 4,
    voxel_size: float = None,
) -> tuple:
    """
    多阶段外参自动配准（参考雷达位姿在配准过程中保持不变）。

    阶段 1 — 初对齐：每台雷达对「参考雷达 + 已对齐雷达」做 ICP，选 RMSE 最小的目标。
    阶段 2 — 全局迭代：多轮将每台雷达对「除自身外所有雷达的融合点云」做 ICP，逐步收紧对应距离直至收敛。
    终检 — 输出各雷达相对融合图的 RMSE / 内点率。

    ref_lid: 配准目标（参考）雷达；为 None 时尝试 lidar_id=1，否则取列表首项。

    返回 (新 offsets, ref_lidar_id, reports, summary_text)。
    """
    if ref_lid is None:
        try:
            ref_lid = resolve_lidar1_id(lidar_ids)
        except ValueError:
            if not lidar_ids:
                raise ValueError("点云中无雷达")
            ref_lid = lidar_ids[0]
    elif not any(same_lidar_id(ref_lid, lid) for lid in lidar_ids):
        raise ValueError(
            "参考雷达 {} 不在当前帧雷达列表中: {}".format(
                lidar_id_to_str(ref_lid), lidar_ids
            )
        )
    for lid in lidar_ids:
        if same_lidar_id(lid, ref_lid):
            ref_lid = lid
            break

    pts_full = structured_to_points_array(structured, fields)
    ref_off_locked = list(get_offset_for_lidar(ref_lid, offsets))
    ref_xyz = _lidar_xyz_transformed(pts_full, ref_lid, ref_off_locked)
    if len(ref_xyz) < 10:
        raise ValueError(
            "参考雷达 {} 点数过少（<10）".format(lidar_id_to_str(ref_lid))
        )

    if voxel_size is None:
        voxel_size = auto_voxel_size(ref_xyz)

    new_offsets = {lid: list(get_offset_for_lidar(lid, offsets)) for lid in lidar_ids}
    new_offsets[ref_lid] = list(ref_off_locked)

    reports = []
    summary_lines = [
        "方案：参考雷达固定 → 最佳配对初对齐 → 多轮全局融合精配准",
        "参考雷达: {}".format(lidar_id_to_str(ref_lid)),
    ]

    reports.append(
        {
            "lidar_id": ref_lid,
            "stage": "anchor",
            "role": "reference",
            "rmse": 0.0,
            "converged": True,
            "target": None,
            "message": "位姿锁定不变",
        }
    )

    sorted_others = _sort_lidar_ids_numeric(lidar_ids, ref_lid)
    aligned_pool = [ref_lid]

    # ---------- 阶段 1：最佳配对初对齐 ----------
    for lid in sorted_others:
        src_raw = pts_full[mask_lidar(pts_full, lid)][:, :3]
        if len(src_raw) < 10:
            reports.append(
                {
                    "lidar_id": lid,
                    "stage": "pairwise",
                    "role": "skipped",
                    "rmse": None,
                    "converged": False,
                    "target": None,
                    "message": "点数过少",
                }
            )
            continue

        old_off = get_offset_for_lidar(lid, new_offsets)
        src_xyz = _lidar_xyz_transformed(pts_full, lid, old_off)

        candidates = [ref_lid] + [
            j for j in aligned_pool if not same_lidar_id(j, lid)
        ]
        best_rmse = np.inf
        best_t = None
        best_tgt = None
        best_conv = False

        for tgt_lid in candidates:
            tgt_xyz = _lidar_xyz_transformed(
                pts_full, tgt_lid, get_offset_for_lidar(tgt_lid, new_offsets)
            )
            t_icp, rmse, conv = _run_icp_pair(
                src_xyz,
                tgt_xyz,
                voxel_size,
                max_points_per_lidar,
                max_iterations,
                max_correspondence_distance,
            )
            if t_icp is None or not np.isfinite(rmse):
                continue
            if rmse < best_rmse:
                best_rmse = rmse
                best_t = t_icp
                best_tgt = tgt_lid
                best_conv = conv

        if best_t is None:
            reports.append(
                {
                    "lidar_id": lid,
                    "stage": "pairwise",
                    "role": "failed",
                    "rmse": None,
                    "converged": False,
                    "target": None,
                    "message": "未找到有效配对",
                }
            )
            aligned_pool.append(lid)
            continue

        new_offsets[lid] = _offset_apply_icp_delta(old_off, best_t)
        aligned_pool.append(lid)
        reports.append(
            {
                "lidar_id": lid,
                "stage": "pairwise",
                "role": "aligned",
                "rmse": best_rmse,
                "converged": best_conv,
                "target": lidar_id_to_str(best_tgt),
                "message": "初对齐目标: 雷达 {}".format(lidar_id_to_str(best_tgt)),
            }
        )

    # ---------- 阶段 2：多轮全局融合精配准 ----------
    dist_schedule = [
        max_correspondence_distance,
        max_correspondence_distance * 0.75,
        max_correspondence_distance * 0.55,
        max_correspondence_distance * 0.4,
    ]
    for rnd in range(max(1, int(global_refine_rounds))):
        max_delta = 0.0
        max_dist = dist_schedule[min(rnd, len(dist_schedule) - 1)]
        for lid in sorted_others:
            src_raw = pts_full[mask_lidar(pts_full, lid)][:, :3]
            if len(src_raw) < 10:
                continue
            target = _build_merged_target_excluding(
                pts_full,
                new_offsets,
                lidar_ids,
                lid,
                voxel_size,
                max_points_per_lidar,
            )
            if target is None or len(target) < 10:
                continue
            old_off = get_offset_for_lidar(lid, new_offsets)
            src_xyz = _lidar_xyz_transformed(pts_full, lid, old_off)
            t_icp, rmse, conv = _run_icp_pair(
                src_xyz,
                target,
                voxel_size,
                max_points_per_lidar,
                max_iterations,
                max_dist,
            )
            if t_icp is None:
                continue
            new_off = _offset_apply_icp_delta(old_off, t_icp)
            max_delta = max(max_delta, _offset_delta_norm(old_off, new_off))
            new_offsets[lid] = new_off
            reports.append(
                {
                    "lidar_id": lid,
                    "stage": "global_r{}".format(rnd + 1),
                    "role": "refined",
                    "rmse": rmse,
                    "converged": conv,
                    "target": "融合点云(除自身)",
                    "message": "对应距离上限 {:.2f}m".format(max_dist),
                }
            )
        summary_lines.append(
            "全局轮次 {}: 最大参数变化 {:.4f}".format(rnd + 1, max_delta)
        )
        if max_delta < 0.02:
            summary_lines.append("全局迭代已收敛，提前结束")
            break

    # 锚定：参考雷达保持配准前位姿（防止数值漂移写入）
    new_offsets[ref_lid] = list(ref_off_locked)

    # 各雷达最终 RMSE（相对融合图）
    for lid in sorted_others:
        target = _build_merged_target_excluding(
            pts_full, new_offsets, lidar_ids, lid, voxel_size, max_points_per_lidar
        )
        src_xyz = _lidar_xyz_transformed(
            pts_full, lid, get_offset_for_lidar(lid, new_offsets)
        )
        if target is None or len(src_xyz) < 10:
            continue
        src_ds = _prepare_icp_cloud(src_xyz, voxel_size, max_points_per_lidar)
        tgt_ds = _prepare_icp_cloud(target, voxel_size, max_points_per_lidar)
        rmse, ratio, _ = icp_fitness(
            src_ds, tgt_ds, max_correspondence_distance=max_correspondence_distance * 0.5
        )
        summary_lines.append(
            "雷达 {} 终检 RMSE={:.4f}m 内点率={:.1%}".format(
                lidar_id_to_str(lid), rmse, ratio
            )
        )

    return new_offsets, ref_lid, reports, "\n".join(summary_lines)


def icp_align_offsets_to_lidar1(
    structured,
    fields,
    offsets: dict,
    lidar_ids: list,
    **kwargs,
) -> tuple:
    """兼容旧接口：仅返回 offsets, ref_lid, reports（无 summary）。"""
    new_offsets, ref_lid, reports, _ = icp_multi_refine_align(
        structured, fields, offsets, lidar_ids, global_refine_rounds=1, **kwargs
    )
    return new_offsets, ref_lid, reports


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


def offsets_for_export(offsets: dict, lidar_ids: list) -> dict:
    out = {}
    for lid in lidar_ids:
        key = str(int(lid)) if float(lid).is_integer() else str(lid)
        out[key] = [round(v, 6) for v in offsets[lid]]
    return out


def export_offsets_json(path, pcd_path, offsets, lidar_ids, ref_lidar_id):
    payload = {
        "source_pcd": pcd_path,
        "description": "相对加载时点云坐标系的增量变换; 参考雷达在 ICP 配准时保持不变",
        "reference_lidar_id": (
            None if ref_lidar_id is None else str(ref_lidar_id)
        ),
        "euler_seq": "xyz",
        "degrees": True,
        "offsets": offsets_for_export(offsets, lidar_ids),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_offsets_json(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return dict(data["offsets"])


def _resolve_pcd_metadata(metadata, source_pcd_path):
    if metadata is not None:
        return metadata
    if source_pcd_path:
        from utils.load_pcd import get_metadata_from_pcd_file

        return get_metadata_from_pcd_file(source_pcd_path)
    raise ValueError("保存 PCD 需要 metadata 或有效的 source_pcd_path")


def save_corrected_structured_with_wata(
    corrected_structured, fields, source_pcd_path, save_path, metadata=None
):
    """
    保存校正后 PCD。优先用内存中的 structured + metadata，避免 wata 重读源文件；
    wata 在 NumPy 2 下可能失败，则回退到本地 save_structured_points。
    """
    md = _resolve_pcd_metadata(metadata, source_pcd_path)
    data_type = md.get("data", "binary_compressed")
    corrected_structured = np.ascontiguousarray(corrected_structured)

    try:
        import wata

        wata.PointCloudProcess.save_pcd_from_structured_points(
            corrected_structured, save_path, type=data_type
        )
        return
    except Exception:
        pass

    from utils.load_pcd import save_structured_points

    save_structured_points(corrected_structured, md, save_path)


def save_corrected_pcd_with_wata(
    corrected_points, source_pcd_path, save_path, metadata=None, fields=None
):
    """从 (N,C) 数组保存；需 fields 与 metadata 以构造 structured。"""
    from utils.load_pcd import _build_dtype

    md = _resolve_pcd_metadata(metadata, source_pcd_path)
    if fields is None:
        fields = md["fields"]
    valid = [f for f in fields if f != "_"]
    dtype = _build_dtype(md)
    structured = np.zeros(corrected_points.shape[0], dtype=dtype)
    for i, fname in enumerate(valid):
        if fname in structured.dtype.names:
            structured[fname] = corrected_points[:, i]
    save_corrected_structured_with_wata(
        structured, fields, source_pcd_path, save_path, metadata=md
    )
