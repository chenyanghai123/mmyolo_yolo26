"""COCO-style mAP computation (minimal, pure-Python)."""
import numpy as np


def _iou_matrix(boxes1, boxes2):
    """Compute IoU matrix (N, M) between xyxy boxes."""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-7)


def compute_map(all_dets, all_gts, num_classes, iou_thr=0.5, area_mode="area"):
    """Compute COCO-style mAP at a single IoU threshold (simplified).

    Args:
        all_dets: list of (N, 6) per image [x1,y1,x2,y2,score,class]
        all_gts: list of (M, 5) per image [x1,y1,x2,y2,class]
        num_classes: int
        iou_thr: float
        area_mode: ignored (kept for compat)
    Returns:
        dict with 'map' and per-class 'ap'
    """
    aps = []
    per_class_ap = {}
    for c in range(num_classes):
        # Collect dets and gts for this class
        det_scores = []
        det_imgs = []
        det_boxes = []
        gt_per_img = []
        n_pos = 0
        for i, (dets, gts) in enumerate(zip(all_dets, all_gts)):
            d_c = dets[dets[:, 5] == c] if len(dets) else np.zeros((0, 6))
            g_c = gts[gts[:, 4] == c] if len(gts) else np.zeros((0, 5))
            det_scores.extend(d_c[:, 4].tolist())
            det_imgs.extend([i] * len(d_c))
            det_boxes.append(d_c[:, :4])
            gt_per_img.append(g_c)
            n_pos += len(g_c)
        if n_pos == 0:
            per_class_ap[c] = float("nan")
            continue
        det_scores = np.array(det_scores)
        order = det_scores.argsort()[::-1]
        det_boxes_all = np.concatenate(det_boxes, axis=0) if det_boxes else np.zeros((0, 4))
        det_imgs_arr = np.array(det_imgs)

        used = [np.zeros(len(g), dtype=bool) for g in gt_per_img]
        tp = np.zeros(len(order))
        fp = np.zeros(len(order))
        for rank, idx in enumerate(order):
            img_id = det_imgs_arr[idx]
            box = det_boxes_all[idx][None, :]
            gts = gt_per_img[img_id]
            if len(gts) == 0:
                fp[rank] = 1
                continue
            ious = _iou_matrix(box, gts[:, :4])[0]
            best_iou = ious.max()
            best_j = ious.argmax()
            if best_iou >= iou_thr and not used[img_id][best_j]:
                tp[rank] = 1
                used[img_id][best_j] = True
            else:
                fp[rank] = 1
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(n_pos, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-7)
        # 101-point interpolation
        rec_thr = np.linspace(0, 1, 101)
        ap = 0.0
        for r in rec_thr:
            mask = recall >= r
            p = precision[mask].max() if mask.any() else 0.0
            ap += p
        ap /= 101
        aps.append(ap)
        per_class_ap[c] = ap
    mAP = float(np.mean(aps)) if aps else 0.0
    return {"map": mAP, "per_class_ap": per_class_ap}
