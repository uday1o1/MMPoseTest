import os, time, csv, argparse, math
import numpy as np
import cv2
from mmpose.apis import MMPoseInferencer

# local files
POSE = 'models/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py'
POSE_WTS = 'models/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth'
DET  = 'models/rtmdet_nano_320-8xb32_coco-person.py'
DEVICE = 'cpu'

# thresholds
HIP_MIN, HIP_MAX = 60.0, 120.0          # avg hip angle window
KNEEHIP_MAX = 0.20                      # |y_knee - y_hip| / body_height
KNEE_TOE_MAX = 0.10                     # |x_knee - x_toe| / hip_width
MIN_SCORE = 0.40

# webcam
CAM_DEFAULT = 0
FRAME_W, FRAME_H = 960, 540

# Halpe-26 fallback
HALPE26_NAMES = [
    "Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow",
    "LWrist","RWrist","LHip","RHip","LKnee","Rknee","LAnkle","RAnkle","Head","Neck",
    "Hip","LBigToe","RBigToe","LSmallToe","RSmallToe","LHeel","RHeel"
]
KP_NAMES = HALPE26_NAMES
NAMES_READY = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=str, default=None, help='an image path; if set, webcam is skipped')
    p.add_argument('--save', type=str, default='media/outputs/currentVis.jpg', help='output image path (auto-numbered)')
    p.add_argument('--csv', action='store_true', help='save CSV logs of keypoints')
    p.add_argument('--cam', type=int, default=CAM_DEFAULT, help='webcam index')
    p.add_argument('--width', type=int, default=FRAME_W, help='webcam width')
    p.add_argument('--height', type=int, default=FRAME_H, help='webcam height')
    return p.parse_args()

# helpers
def angle_deg(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    den = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = np.clip(np.dot(ba, bc) / den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def pick_best_instance(instances):
    if not instances: return None
    def avg_score(inst):
        s = inst['keypoint_scores']
        return float(np.mean(s)) if len(s) else 0.0
    return max(instances, key=avg_score)

def label_points(names, kpts, scores, min_score=0.0):
    out = {}
    for i in range(kpts.shape[0]):
        n = names[i] if names and i < len(names) else str(i)
        x, y = float(kpts[i,0]), float(kpts[i,1])
        s = float(scores[i])
        if s >= min_score:
            out[n] = {'x': x, 'y': y, 'score': s}
    return out

def _pick_key(points, *cands):
    if not points: return None
    lower = {k.lower(): k for k in points.keys()}
    for c in cands:
        k = lower.get(c.lower())
        if k: return k
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
        "L_HEEL": ("LHeel","left_heel"),
        "R_HEEL": ("RHeel","right_heel"),
        "NECK": ("Neck","neck"),
        "HEAD": ("Head","head"),
        "HIP_C": ("Hip","hip","pelvis")
    }
    resolved = {}
    for key, cands in CANON.items():
        name = _pick_key(points, *cands)
        if name: resolved[key] = name
    return resolved

def foot_rotation_deg(points, canon):
    degs = []
    for heel_key, toe_key in (("L_HEEL","L_BIGTOE"), ("R_HEEL","R_BIGTOE")):
        if heel_key in canon and toe_key in canon:
            h = points[canon[heel_key]]; t = points[canon[toe_key]]
            dx, dy = (t['x'] - h['x']), (t['y'] - h['y'])
            if abs(dx) + abs(dy) > 1e-6:
                degs.append(math.degrees(math.atan2(dy, dx)))
    if not degs: return 0.0
    return float(sum(degs)/len(degs))

def rotate_points(points, M):
    out = {}
    for k, v in points.items():
        x, y = v['x'], v['y']
        nx = M[0,0]*x + M[0,1]*y + M[0,2]
        ny = M[1,0]*x + M[1,1]*y + M[1,2]
        out[k] = {'x': float(nx), 'y': float(ny), 'score': v['score']}
    return out

def hip_angles_avg(points, canon):
    la = ra = None
    if all(k in canon for k in ("L_KNE","L_HIP","L_SHO")):
        A = points[canon["L_KNE"]]; B = points[canon["L_HIP"]]; C = points[canon["L_SHO"]]
        la = angle_deg((A['x'],A['y']), (B['x'],B['y']), (C['x'],C['y']))
    if all(k in canon for k in ("R_KNE","R_HIP","R_SHO")):
        A = points[canon["R_KNE"]]; B = points[canon["R_HIP"]]; C = points[canon["R_SHO"]]
        ra = angle_deg((A['x'],A['y']), (B['x'],B['y']), (C['x'],C['y']))
    if la is None and ra is None: return None, None, None
    avg = la if ra is None else ra if la is None else 0.5*(la+ra)
    return avg, la, ra

def extract_instances_from_output(out):
    global KP_NAMES, NAMES_READY
    preds = out.get('predictions', [])
    batch_list = preds if isinstance(preds, list) else [preds]
    ds_list = []
    for item in batch_list:
        if isinstance(item, list): ds_list.extend(item)
        elif item is not None: ds_list.append(item)
    if not NAMES_READY:
        for ds in ds_list:
            if hasattr(ds, 'metainfo'):
                meta = getattr(ds, 'metainfo', {})
                dsmeta = meta.get('dataset_meta', {}) if isinstance(meta, dict) else {}
                names = dsmeta.get('keypoint_names') or dsmeta.get('keypoint_id2name')
                if isinstance(names, dict): names = [names[i] for i in sorted(names)]
                if isinstance(names, (list, tuple)) and len(names) >= 17:
                    KP_NAMES = list(names); NAMES_READY = True; break
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
            else:
                instances.append({'keypoints': k, 'keypoint_scores': s})
        elif isinstance(ds, dict) and 'keypoints' in ds:
            k = np.array(ds['keypoints']); s = np.array(ds['keypoint_scores'])
            if k.ndim == 3:
                for i in range(k.shape[0]):
                    instances.append({'keypoints': k[i], 'keypoint_scores': s[i]})
            else:
                instances.append({'keypoints': k, 'keypoint_scores': s})
    return instances

def build_inferencer():
    for p, label in [(POSE,'POSE config'), (POSE_WTS,'POSE weights'), (DET,'DET config')]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} not found: {p}")
    return MMPoseInferencer(pose2d=POSE, pose2d_weights=POSE_WTS, det_model=DET, device=DEVICE)

def ensure_dir(path):
    d = os.path.dirname(path) or '.'
    os.makedirs(d, exist_ok=True)

def unique_save_path(path):
    ensure_dir(path)
    base, ext = os.path.splitext(path)
    if not os.path.exists(path): return path
    n = 1
    while True:
        cand = f"{base}_{n}{ext}"
        if not os.path.exists(cand): return cand
        n += 1

def evaluate_and_draw(frame_bgr, out, csv_writer=None, do_calibrate=True):
    # use MMPose’s visualization so keypoints + skeleton are shown
    overlay = frame_bgr
    if out.get('visualization'):
        vis_rgb = out['visualization'][0]
        overlay = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    instances = extract_instances_from_output(out)
    best = pick_best_instance(instances)
    if best is None:
        cv2.putText(overlay, "No person detected", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,0,255), 3)
        return overlay

    # points for rules
    kpts = best['keypoints']; scores = best['keypoint_scores']
    points = label_points(KP_NAMES, kpts, scores, min_score=MIN_SCORE)
    canon = resolve_halpe_canon(points)

    # camera roll calibration from feet
    H, W = overlay.shape[:2]
    rot_deg = 0.0
    if do_calibrate:
        rot_deg = -foot_rotation_deg(points, canon)
        M = cv2.getRotationMatrix2D((W/2.0, H/2.0), rot_deg, 1.0)
        overlay = cv2.warpAffine(overlay, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        points = rotate_points(points, M)       # keep rules consistent after rotation
        canon = resolve_halpe_canon(points)

    # Condition 1: avg hip angle
    hip_avg, hip_L, hip_R = hip_angles_avg(points, canon)

    # normalizers
    hip_w = 1.0
    if all(k in canon for k in ("L_HIP","R_HIP")):
        hip_w = abs(points[canon["R_HIP"]]['x'] - points[canon["L_HIP"]]['x']) + 1e-6

    body_h = 1.0
    if all(k in canon for k in ("L_SHO","R_SHO","L_ANK","R_ANK")):
        y_sh = 0.5*(points[canon["L_SHO"]]['y'] + points[canon["R_SHO"]]['y'])
        y_an = 0.5*(points[canon["L_ANK"]]['y'] + points[canon["R_ANK"]]['y'])
        body_h = abs(y_an - y_sh) + 1e-6

    # Condition 2: knee-hip vertical difference
    dyL_n = dyR_n = None
    if all(k in canon for k in ("L_KNE","L_HIP")):
        dyL_n = abs(points[canon["L_KNE"]]['y'] - points[canon["L_HIP"]]['y']) / body_h
    if all(k in canon for k in ("R_KNE","R_HIP")):
        dyR_n = abs(points[canon["R_KNE"]]['y'] - points[canon["R_HIP"]]['y']) / body_h

    # Condition 3: knee-toe lateral difference
    L_toe = canon.get("L_BIGTOE") or canon.get("L_ANK")
    R_toe = canon.get("R_BIGTOE") or canon.get("R_ANK")
    dxL = dxR = None
    if all(k in canon for k in ("L_KNE","R_KNE")) and L_toe and R_toe:
        dxL = abs(points[canon["L_KNE"]]['x'] - points[L_toe]['x']) / hip_w
        dxR = abs(points[canon["R_KNE"]]['x'] - points[R_toe]['x']) / hip_w

    # checks
    ok1 = (hip_avg is not None) and (HIP_MIN <= hip_avg <= HIP_MAX)
    ok2 = (dyL_n is not None and dyR_n is not None and dyL_n <= KNEEHIP_MAX and dyR_n <= KNEEHIP_MAX)
    ok3 = (dxL is not None and dxR is not None and dxL <= KNEE_TOE_MAX and dxR <= KNEE_TOE_MAX)
    ok_all = ok1 and ok2 and ok3

    # big status banner
    color = (0,255,0) if ok_all else (0,0,255)
    cv2.putText(overlay, "FORM OK" if ok_all else "FORM ISSUE",
                (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 3)

    # guidance lines
    tips = []
    if hip_avg is not None:
        if hip_avg < HIP_MIN: tips.append("raise hips a bit")
        elif hip_avg > HIP_MAX: tips.append("lower hips a bit")
    if (dyL_n is not None and dyR_n is not None) and (dyL_n > KNEEHIP_MAX or dyR_n > KNEEHIP_MAX):
        tips.append("keep thighs horizontal")
    if (dxL is not None and dxR is not None) and (dxL > KNEE_TOE_MAX or dxR > KNEE_TOE_MAX):
        tips.append("knees behind toes")

    y0 = 74
    for g in tips[:3]:
        cv2.putText(overlay, f"Tip: {g}", (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,255,255), 2)
        y0 += 28

    if csv_writer is not None:
        ts = int(time.time() * 1000)
        for j, d in points.items():
            csv_writer.writerow([ts, 0, j, d['x'], d['y'], d['score']])

    return overlay

def run_on_image(infer, img_path, save_path, save_csv):
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError(f"cannot read image: {img_path}")
    out = next(infer(img, return_vis=True, return_datasamples=True, show=False))
    overlay = evaluate_and_draw(img, out, None)
    final_path = unique_save_path(save_path)
    cv2.imwrite(final_path, overlay)
    print(f"saved -> {final_path}")

def run_on_webcam(infer, cam, width, height, save_csv):
    csv_writer = None; csv_file = None
    if save_csv:
        os.makedirs('media/outputs', exist_ok=True)
        csv_file = open('media/outputs/keypoints_log.csv', 'a', newline='')
        csv_writer = csv.writer(csv_file)
        if os.stat('media/outputs/keypoints_log.csv').st_size == 0:
            csv_writer.writerow(['ts_ms','person','joint','x','y','score'])

    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("cannot open camera; check permissions or index")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            out = next(infer(frame, return_vis=True, return_datasamples=True, show=False))
            overlay = evaluate_and_draw(frame, out, csv_writer)
            cv2.imshow("press q to quit", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cap.release(); cv2.destroyAllWindows()
        if csv_file: csv_file.close()

if __name__ == "__main__":
    args = parse_args()
    infer = build_inferencer()
    if args.image:
        os.makedirs('media/outputs', exist_ok=True)
        run_on_image(infer, args.image, args.save, args.csv)
    else:
        run_on_webcam(infer, args.cam, args.width, args.height, args.csv)
