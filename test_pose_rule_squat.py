import os, time, csv, argparse
import numpy as np
import cv2
from mmpose.apis import MMPoseInferencer

# config
POSE = 'models/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py'
POSE_WTS = 'models/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth'
DET = 'models/rtmdet_nano_320-8xb32_coco-person.py'
DEVICE = 'cpu'
MIN_SCORE = 0.40
CAM_DEFAULT = 0
FRAME_W, FRAME_H = 960, 540

# thresholds
KNEE_MIN, KNEE_MAX = 60.0, 120.0
DX_MAX = 0.10
KNEEHIP_MAX = 0.20

# fallback names for Halpe-26
HALPE26_NAMES = [
    "Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow",
    "LWrist","RWrist","LHip","RHip","LKnee","Rknee","LAnkle","RAnkle","Head","Neck",
    "Hip","LBigToe","RBigToe","LSmallToe","RSmallToe","LHeel","RHeel"
]
KP_NAMES = HALPE26_NAMES
NAMES_READY = False

# helpers
def angle_deg(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    den = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = np.clip(np.dot(ba, bc) / den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def pick_best_instance(instances):
    if not instances:
        return None
    def avg_score(inst):
        s = inst['keypoint_scores']
        return float(np.mean(s)) if len(s) else 0.0
    return max(instances, key=avg_score)

def label_points(names, kpts, scores, min_score=0.0):
    out = {}
    K = kpts.shape[0]
    for i in range(K):
        n = names[i] if names and i < len(names) else str(i)
        x, y = float(kpts[i, 0]), float(kpts[i, 1])
        s = float(scores[i])
        if s >= min_score:
            out[n] = {'x': x, 'y': y, 'score': s}
    return out

def _pick_key(points, *cands):
    if not points:
        return None
    lower = {k.lower(): k for k in points.keys()}
    for c in cands:
        k = lower.get(c.lower())
        if k:
            return k
    return None

def resolve_halpe_canon(points):
    CANON = {
        "L_SHO": ("LShoulder","left_shoulder","L_Shoulder"),
        "R_SHO": ("RShoulder","right_shoulder","R_Shoulder"),
        "L_HIP": ("LHip","left_hip","L_Hip"),
        "R_HIP": ("RHip","right_hip","R_Hip"),
        "L_KNE": ("LKnee","left_knee","L_Knee"),
        "R_KNE": ("Rknee","right_knee","R_Knee"),
        "L_ANK": ("LAnkle","left_ankle","L_Ankle"),
        "R_ANK": ("RAnkle","right_ankle","R_Ankle"),
        "L_BIGTOE": ("LBigToe","left_big_toe","left_foot_index"),
        "R_BIGTOE": ("RBigToe","right_big_toe","right_foot_index"),
        "L_SMTOE": ("LSmallToe","left_small_toe"),
        "R_SMTOE": ("RSmallToe","right_small_toe"),
        "L_HEEL": ("LHeel","left_heel"),
        "R_HEEL": ("RHeel","right_heel"),
        "NECK": ("Neck","neck"),
        "HEAD": ("Head","head"),
        "HIP_C": ("Hip","hip","pelvis")
    }
    resolved = {}
    for key, cands in CANON.items():
        name = _pick_key(points, *cands)
        if name:
            resolved[key] = name
    return resolved

def squat_knee_angle(points, canon):
    left_ok = all(k in canon for k in ("L_HIP","L_KNE","L_ANK"))
    right_ok = all(k in canon for k in ("R_HIP","R_KNE","R_ANK"))
    if not (left_ok or right_ok):
        return None
    if left_ok:
        H, K, A = canon["L_HIP"], canon["L_KNE"], canon["L_ANK"]
    else:
        H, K, A = canon["R_HIP"], canon["R_KNE"], canon["R_ANK"]
    hip = (points[H]['x'], points[H]['y'])
    knee = (points[K]['x'], points[K]['y'])
    ank = (points[A]['x'], points[A]['y'])
    return angle_deg(hip, knee, ank)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=str, default=None, help='path to an image; if set, webcam is skipped')
    p.add_argument('--save', type=str, default='media/outputs/currentVis.jpg', help='output image path (auto-numbered)')
    p.add_argument('--csv', action='store_true', help='save CSV logs')
    p.add_argument('--cam', type=int, default=CAM_DEFAULT, help='webcam index')
    p.add_argument('--width', type=int, default=FRAME_W, help='webcam width')
    p.add_argument('--height', type=int, default=FRAME_H, help='webcam height')
    return p.parse_args()

def build_inferencer():
    infer = MMPoseInferencer(
        pose2d=POSE,
        pose2d_weights=POSE_WTS,
        det_model=DET,
        device=DEVICE
    )
    print(f"Loaded ({POSE}); detector={DET}; device={DEVICE}")
    return infer

def extract_instances_from_output(out):
    global KP_NAMES, NAMES_READY
    preds = out.get('predictions', [])
    batch_list = preds if isinstance(preds, list) else [preds]
    ds_list = []
    for item in batch_list:
        if isinstance(item, list):
            ds_list.extend(item)
        elif item is not None:
            ds_list.append(item)
    if not NAMES_READY:
        for ds in ds_list:
            if hasattr(ds, 'metainfo'):
                meta = getattr(ds, 'metainfo', {})
                dsmeta = meta.get('dataset_meta', {}) if isinstance(meta, dict) else {}
                names = dsmeta.get('keypoint_names') or dsmeta.get('keypoint_id2name')
                if isinstance(names, dict):
                    names = [names[i] for i in sorted(names)]
                if isinstance(names, (list, tuple)) and len(names) >= 17:
                    KP_NAMES = list(names)
                    NAMES_READY = True
                    break
    instances = []
    for ds in ds_list:
        if hasattr(ds, 'pred_instances'):
            k = ds.pred_instances.keypoints
            s = ds.pred_instances.keypoint_scores
            k = k.detach().cpu().numpy() if hasattr(k, 'detach') else np.array(k)
            s = s.detach().cpu().numpy() if hasattr(s, 'detach') else np.array(s)
            if k.ndim == 3:
                for i in range(k.shape[0]):
                    instances.append({'keypoints': k[i], 'keypoint_scores': s[i]})
            elif k.ndim == 2:
                instances.append({'keypoints': k, 'keypoint_scores': s})
        elif isinstance(ds, dict) and 'keypoints' in ds:
            k = np.array(ds['keypoints'])
            s = np.array(ds['keypoint_scores'])
            if k.ndim == 3:
                for i in range(k.shape[0]):
                    instances.append({'keypoints': k[i], 'keypoint_scores': s[i]})
            else:
                instances.append({'keypoints': k, 'keypoint_scores': s})
    return instances

def evaluate_and_draw(overlay, instances, csv_writer=None):
    ok_all = False
    best = pick_best_instance(instances)
    if best is None:
        return overlay, ok_all
    kpts = best['keypoints']
    scores = best['keypoint_scores']
    points = label_points(KP_NAMES, kpts, scores, min_score=MIN_SCORE)
    canon = resolve_halpe_canon(points)

    knee_ang = squat_knee_angle(points, canon)
    if knee_ang is not None:
        cv2.putText(overlay, f"Knee angle: {knee_ang:.1f} deg",
                    (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    hip_w = 1.0
    if all(k in canon for k in ("L_HIP","R_HIP")):
        hip_w = abs(points[canon["R_HIP"]]['x'] - points[canon["L_HIP"]]['x']) + 1e-6

    L_toe = canon.get("L_BIGTOE") or canon.get("L_ANK")
    R_toe = canon.get("R_BIGTOE") or canon.get("R_ANK")
    dxL = dxR = None
    if all(k in canon for k in ("L_KNE","R_KNE")) and L_toe and R_toe:
        dxL = abs(points[canon["L_KNE"]]['x'] - points[L_toe]['x']) / hip_w
        dxR = abs(points[canon["R_KNE"]]['x'] - points[R_toe]['x']) / hip_w
        cv2.putText(overlay, f"Knee/Toe dx L:{dxL:.3f} R:{dxR:.3f}",
                    (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    body_h = 1.0
    if all(k in canon for k in ("L_SHO","R_SHO","L_ANK","R_ANK")):
        y_sh = 0.5*(points[canon["L_SHO"]]['y'] + points[canon["R_SHO"]]['y'])
        y_an = 0.5*(points[canon["L_ANK"]]['y'] + points[canon["R_ANK"]]['y'])
        body_h = abs(y_an - y_sh) + 1e-6

    dyL_n = dyR_n = None
    if all(k in canon for k in ("L_KNE","L_HIP")):
        dyL_n = abs(points[canon["L_KNE"]]['y'] - points[canon["L_HIP"]]['y']) / body_h
    if all(k in canon for k in ("R_KNE","R_HIP")):
        dyR_n = abs(points[canon["R_KNE"]]['y'] - points[canon["R_HIP"]]['y']) / body_h
    if (dyL_n is not None) and (dyR_n is not None):
        cv2.putText(overlay, f"dyL:{dyL_n:.3f} dyR:{dyR_n:.3f}",
                    (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    ok_knee = (knee_ang is not None) and (KNEE_MIN <= knee_ang <= KNEE_MAX)
    ok_dx = (dxL is not None and dxR is not None and dxL <= DX_MAX and dxR <= DX_MAX)
    ok_kneehip = (dyL_n is not None and dyR_n is not None and dyL_n <= KNEEHIP_MAX and dyR_n <= KNEEHIP_MAX)
    ok_all = ok_knee and ok_dx and ok_kneehip
    color = (0,255,0) if ok_all else (0,0,255)
    cv2.putText(overlay, "FORM OK" if ok_all else "FORM ISSUE",
                (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if csv_writer is not None:
        ts = int(time.time() * 1000)
        for j, data in points.items():
            csv_writer.writerow([ts, 0, j, data['x'], data['y'], data['score']])

    return overlay, ok_all

# auto-numbered save helpers
def ensure_dir(path):
    d = os.path.dirname(path) or '.'
    os.makedirs(d, exist_ok=True)

def unique_save_path(path):
    ensure_dir(path)
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path
    n = 1
    while True:
        cand = f"{base}_{n}{ext}"
        if not os.path.exists(cand):
            return cand
        n += 1

def run_on_image(inferencer, img_path, save_path, save_csv):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {img_path}")
    out = next(inferencer(img, return_vis=True, return_datasamples=True, show=False))
    overlay = img
    if out.get('visualization'):
        vis_rgb = out['visualization'][0]
        overlay = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    csv_writer = None
    csv_file = None
    if save_csv:
        ensure_dir('media/outputs/keypoints_log.csv')
        csv_path = 'media/outputs/keypoints_log.csv'
        write_header = not os.path.exists(csv_path)
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(['ts_ms', 'person', 'joint', 'x', 'y', 'score'])

    instances = extract_instances_from_output(out)
    overlay, ok_all = evaluate_and_draw(overlay, instances, csv_writer)

    final_path = unique_save_path(save_path)
    cv2.imwrite(final_path, overlay)
    if csv_file:
        csv_file.close()
    print(f"saved visualization -> {final_path}")
    print("result:", "FORM OK" if ok_all else "FORM ISSUE")

def run_on_webcam(inferencer, cam, width, height, save_csv):
    csv_writer = None
    csv_file = None
    if save_csv:
        ensure_dir('media/outputs/keypoints_log.csv')
        csv_path = 'media/outputs/keypoints_log.csv'
        write_header = not os.path.exists(csv_path)
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(['ts_ms', 'person', 'joint', 'x', 'y', 'score'])

    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("cannot open camera; check permissions or try a different index")

    fps = 0.0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            t0 = time.time()
            out = next(inferencer(frame_bgr, return_vis=True, return_datasamples=True, show=False))

            overlay = frame_bgr
            if out.get('visualization'):
                vis_rgb = out['visualization'][0]
                overlay = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

            instances = extract_instances_from_output(out)
            overlay, _ = evaluate_and_draw(overlay, instances, csv_writer)

            dt = time.time() - t0
            fps = 0.9*fps + 0.1*(1.0/dt if dt > 0 else 0.0)
            cv2.putText(overlay, f"FPS: {fps:5.1f}", (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("MMPose (CPU, Halpe-26) – press q to quit", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()

if __name__ == "__main__":
    args = parse_args()
    inferencer = build_inferencer()
    if args.image:
        run_on_image(inferencer, args.image, args.save, args.csv)
    else:
        run_on_webcam(inferencer, args.cam, args.width, args.height, args.csv)
