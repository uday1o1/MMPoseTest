import os, csv, argparse, math
from pathlib import Path
import glob
import numpy as np
import cv2
from mmpose.apis import MMPoseInferencer

# local files
POSE = 'models/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py'
POSE_WTS = 'models/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth'
DET  = 'models/rtmdet_nano_320-8xb32_coco-person.py'
DEVICE = 'cpu'

# thresholds
HIP_MIN, HIP_MAX = 60.0, 120.0
KNEEHIP_MAX = 0.20
KNEE_TOE_MAX = 0.10
MIN_SCORE = 0.40

# webcam
CAM_DEFAULT = 0
FRAME_W, FRAME_H = 960, 540

# colors
BANNER_OK = (0, 220, 0)
BANNER_BAD = (0, 0, 255)
TIP_COLOR  = (255, 255, 255)  # white
TIP_SHADOW = (0, 0, 0)

# rotation smoothing/clamp
ROT_ALPHA = 0.2               # EMA factor
ROT_MAX_ABS = 30.0            # clamp rotation to +/- degrees
_rot_state = {'init': False, 'deg': 0.0}

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
    p.add_argument('--image', type=str, default=None, help='image path; if set, webcam is skipped')
    p.add_argument('--save', type=str, default='media/output/currentVis.jpg', help='output image path (auto-numbered)')
    p.add_argument('--csv', action='store_true', help='save CSV logs of keypoints')
    p.add_argument('--cam', type=int, default=CAM_DEFAULT, help='webcam index')
    p.add_argument('--width', type=int, default=FRAME_W, help='webcam width')
    p.add_argument('--height', type=int, default=FRAME_H, help='webcam height')
    p.add_argument('--no_calib', action='store_true', help='disable foot-based roll calibration')
    return p.parse_args()

# helpers
def angle_deg(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return float('nan')  # undefined angle

    cosang = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def pick_best_instance(instances):
    def avg_score(inst):
        s = inst.get("keypoint_scores", [])
        return np.mean(s) if len(s) else 0.0

    return max(instances, key=avg_score, default=None)

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
    out = {}
    for key, cands in CANON.items():
        name = _pick_key(points, *cands)
        if name: out[key] = name
    return out

def _wrap_to_90(deg):
    # treat foot line as orientation (ignore direction)
    return ((deg + 90.0) % 180.0) - 90.0

def foot_rotation_deg(points, canon):
    degs = []
    for heel_key, toe_key in (("L_HEEL","L_BIGTOE"), ("R_HEEL","R_BIGTOE")):
        if heel_key in canon and toe_key in canon:
            h = points[canon[heel_key]]; t = points[canon[toe_key]]
            dx, dy = (t['x'] - h['x']), (t['y'] - h['y'])
            if abs(dx) + abs(dy) > 1e-6:
                ang = math.degrees(math.atan2(dy, dx))
                degs.append(_wrap_to_90(ang))
    if not degs: return 0.0
    # robust average
    degs.sort()
    mid = len(degs)//2
    med = degs[mid] if len(degs)%2 else 0.5*(degs[mid-1]+degs[mid])
    return float(med)

def rotate_points(points, M):
    out = {}
    for k, v in points.items():
        x, y = v['x'], v['y']
        nx = M[0,0]*x + M[0,1]*y + M[0,2]
        ny = M[1,0]*x + M[1,1]*y + M[1,2]
        out[k] = {'x': float(nx), 'y': float(ny), 'score': v['score']}
    return out

def hip_angles_avg(points, canon, min_score=MIN_SCORE):
    def side_angle(knee, hip, sho):
        # indices must exist in canon and points
        idxs = [canon.get(k) for k in (knee, hip, sho)]
        if any(i is None or i not in points for i in idxs):
            return None

        A, B, C = (points[idxs[0]], points[idxs[1]], points[idxs[2]])

        # require decent confidences
        if min(A.get('score', 0.0), B.get('score', 0.0), C.get('score', 0.0)) < min_score:
            return None

        ang = angle_deg((A['x'], A['y']), (B['x'], B['y']), (C['x'], C['y']))
        # guard NaN/inf from degenerate geometry
        if ang is None or math.isnan(ang) or math.isinf(ang):
            return None
        return float(ang)

    la = side_angle("L_KNE", "L_HIP", "L_SHO")
    ra = side_angle("R_KNE", "R_HIP", "R_SHO")

    if la is None and ra is None:
        return None, None, None

    avg = la if ra is None else ra if la is None else 0.5 * (la + ra)
    return avg, la, ra

def extract_instances_from_output(out):
    """
    Returns: list of dicts with:
      - 'keypoints': (K,2) float32 xy
      - 'keypoint_scores': (K,) float32 in [0,1] (zeros if missing)
      - optional: 'bbox' if present
    Also fills KP_NAMES/NAMES_READY (global) once if found.
    """
    import numpy as np
    global KP_NAMES, NAMES_READY

    preds = out.get('predictions', [])
    batch_list = preds if isinstance(preds, list) else [preds]

    # flatten predictions -> ds_list
    ds_list = []
    for item in batch_list:
        if isinstance(item, list):
            ds_list.extend([x for x in item if x is not None])
        elif item is not None:
            ds_list.append(item)

    # try to set KP_NAMES once
    if not NAMES_READY:
        for ds in ds_list:
            meta = getattr(ds, 'metainfo', None)
            if isinstance(meta, dict):
                dsmeta = meta.get('dataset_meta', {}) if isinstance(meta.get('dataset_meta', {}), dict) else meta
                names = dsmeta.get('keypoint_names') or dsmeta.get('keypoint_id2name')
                if isinstance(names, dict):
                    # assume int keys
                    names = [names[i] for i in sorted(names)]
                if isinstance(names, (list, tuple)) and len(names) >= 17:
                    KP_NAMES = list(names)
                    NAMES_READY = True
                    break

    instances = []

    def to_np(x):
        # torch or np -> np
        if hasattr(x, 'detach'):
            x = x.detach().cpu().numpy()
        else:
            x = np.array(x)
        return x

    def standardize_kps_scores(k, s=None):
        """
        k: (K,2) or (K,3) or (N,K,2/3) already split to single instance before calling.
        s: (K,), (K,1) or None.
        Returns (kxy, scor) with shapes (K,2), (K,)
        """
        k = to_np(k)
        if k.ndim != 2:
            raise ValueError(f"keypoints must be (K,2/3), got {k.shape}")
        # keep xy only (drop visibility if present)
        if k.shape[1] >= 2:
            kxy = k[:, :2].astype(np.float32, copy=False)
        else:
            raise ValueError(f"keypoints last dim < 2: {k.shape}")

        if s is None:
            # try to recover from k[:,2] if it looks like vis
            if k.shape[1] >= 3:
                scor = k[:, 2].astype(np.float32, copy=False)
                # normalize vis {0,1,2} to [0,1]
                mx = (scor.max() if scor.size else 1.0) or 1.0
                if mx > 1.0:  # common COCO vis is {0,1,2}
                    scor = scor / mx
            else:
                scor = np.zeros((k.shape[0],), dtype=np.float32)
        else:
            s = to_np(s).astype(np.float32, copy=False)
            s = s.squeeze()
            if s.ndim != 1 or s.shape[0] != k.shape[0]:
                # shape mismatch -> best effort
                s = np.resize(s, (k.shape[0],)).astype(np.float32)
            scor = s
        return kxy, scor

    for ds in ds_list:
        if hasattr(ds, 'pred_instances'):
            inst = ds.pred_instances
            k = getattr(inst, 'keypoints', None)
            s = getattr(inst, 'keypoint_scores', None)
            b = getattr(inst, 'bboxes', None)

            if k is None:
                continue
            # convert to np
            k = to_np(k)
            s = None if s is None else to_np(s)
            b = None if b is None else to_np(b)

            # handle shapes: (N,K,2/3) or (K,2/3)
            if k.ndim == 3:
                N = k.shape[0]
                for i in range(N):
                    kxy, scor = standardize_kps_scores(k[i], None if s is None else s[i])
                    entry = {'keypoints': kxy, 'keypoint_scores': scor}
                    if b is not None:
                        # b can be (N,4) or (4,)
                        bb = b[i] if b.ndim == 2 else b
                        entry['bbox'] = bb.astype(np.float32, copy=False)
                    instances.append(entry)
            elif k.ndim == 2:
                kxy, scor = standardize_kps_scores(k, s)
                entry = {'keypoints': kxy, 'keypoint_scores': scor}
                if b is not None:
                    entry['bbox'] = (b.astype(np.float32) if b.ndim == 1 else b[0].astype(np.float32))
                instances.append(entry)
        elif isinstance(ds, dict) and 'keypoints' in ds:
            k = ds['keypoints']
            s = ds.get('keypoint_scores', None)
            # handle (N,K,*) or (K,*) dict payloads
            k = to_np(k)
            s = None if s is None else to_np(s)
            if k.ndim == 3:
                for i in range(k.shape[0]):
                    kxy, scor = standardize_kps_scores(k[i], None if s is None else s[i])
                    instances.append({'keypoints': kxy, 'keypoint_scores': scor})
            elif k.ndim == 2:
                kxy, scor = standardize_kps_scores(k, s)
                instances.append({'keypoints': kxy, 'keypoint_scores': scor})

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

def put_text(img, text, org, scale, color, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, TIP_SHADOW, thick+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def evaluate_and_draw(frame_bgr, out, csv_writer=None, do_calibrate=True, debug=True):
    # visualization (safe)
    vis = out.get('visualization')
    if isinstance(vis, (list, tuple)) and len(vis) > 0 and vis[0] is not None:
        overlay = cv2.cvtColor(vis[0], cv2.COLOR_RGB2BGR)
    else:
        overlay = frame_bgr.copy()

    # pick best person
    instances = extract_instances_from_output(out)
    best = pick_best_instance(instances)
    if best is None:
        put_text(overlay, "No person detected", (12, 42), 0.95, BANNER_BAD, 3)
        if debug:
            print("[DBG] no person detected")
        return overlay

    # prepare points
    kpts = best['keypoints']; scores = best['keypoint_scores']
    points = label_points(KP_NAMES, kpts, scores, min_score=MIN_SCORE)
    canon = resolve_halpe_canon(points)

    H, W = overlay.shape[:2]
    # rotation calibration (EMA)
    _ = globals().setdefault('_rot_state', {'init': False, 'deg': 0.0})
    ROT_ALPHA = globals().get('ROT_ALPHA', 0.3)
    ROT_MAX_ABS = globals().get('ROT_MAX_ABS', 25.0)

    rot_deg = 0.0
    if do_calibrate:
        raw = -foot_rotation_deg(points, canon)
        if _rot_state['init']:
            rot_deg = (1.0 - ROT_ALPHA) * _rot_state['deg'] + ROT_ALPHA * raw
        else:
            rot_deg = raw; _rot_state['init'] = True
        rot_deg = max(-ROT_MAX_ABS, min(ROT_MAX_ABS, rot_deg))
        if abs(rot_deg) < 0.5:  # deadband
            rot_deg = 0.0
        _rot_state['deg'] = rot_deg

        M = cv2.getRotationMatrix2D((W/2.0, H/2.0), rot_deg, 1.0)
        overlay = cv2.warpAffine(overlay, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        points = rotate_points(points, M)
        canon = resolve_halpe_canon(points)

    def _has_points(*keys):
        return all((k in canon) and (canon[k] in points) for k in keys)

    # ---- Condition 1: Hip angle between HIP_MIN and HIP_MAX (avg L/R) ----
    hip_avg, hip_L, hip_R = hip_angles_avg(points, canon)
    ok1 = (hip_avg is not None) and (HIP_MIN <= hip_avg <= HIP_MAX)

    # Normalizers
    hip_w = 1.0
    if _has_points("L_HIP", "R_HIP"):
        hip_w = abs(points[canon["R_HIP"]]['x'] - points[canon["L_HIP"]]['x']) + 1e-6

    body_h = 1.0
    if _has_points("L_SHO", "R_SHO", "L_ANK", "R_ANK"):
        y_sh = 0.5*(points[canon["L_SHO"]]['y'] + points[canon["R_SHO"]]['y'])
        y_an = 0.5*(points[canon["L_ANK"]]['y'] + points[canon["R_ANK"]]['y'])
        body_h = abs(y_an - y_sh) + 1e-6

    # ---- Condition 2: |knee_y - hip_y| / body_h <= KNEEHIP_MAX ----
    dyL_n = dyR_n = None
    if _has_points("L_KNE","L_HIP"):
        dyL_n = abs(points[canon["L_KNE"]]['y'] - points[canon["L_HIP"]]['y']) / body_h
    if _has_points("R_KNE","R_HIP"):
        dyR_n = abs(points[canon["R_KNE"]]['y'] - points[canon["R_HIP"]]['y']) / body_h
    ok2 = (dyL_n is not None and dyR_n is not None and dyL_n <= KNEEHIP_MAX and dyR_n <= KNEEHIP_MAX)

    # ---- Condition 3 (directional): knee should NOT be AHEAD of toe by more than KNEE_TOE_MAX ----
    # get toe indices with safe fallback
    L_toe = canon["L_BIGTOE"] if "L_BIGTOE" in canon else canon.get("L_ANK")
    R_toe = canon["R_BIGTOE"] if "R_BIGTOE" in canon else canon.get("R_ANK")

    dxL = dxR = None
    cond3_checks = []  # collect booleans for sides we CAN evaluate

    # left side
    if ("L_KNE" in canon) and isinstance(L_toe, int) and (canon["L_KNE"] in points) and (L_toe in points):
        dxL = (points[canon["L_KNE"]]['x'] - points[L_toe]['x']) / hip_w
        cond3_checks.append(dxL <= KNEE_TOE_MAX)

    # right side
    if ("R_KNE" in canon) and isinstance(R_toe, int) and (canon["R_KNE"] in points) and (R_toe in points):
        dxR = (points[canon["R_KNE"]]['x'] - points[R_toe]['x']) / hip_w
        cond3_checks.append(dxR <= KNEE_TOE_MAX)

    # No data => do NOT fail the rule (avoid false "FORM ISSUE")
    ok3 = all(cond3_checks) if cond3_checks else True

    ok_all = ok1 and ok2 and ok3
    put_text(overlay, "FORM OK" if ok_all else "FORM ISSUE",
             (12, 42), 0.95, BANNER_OK if ok_all else BANNER_BAD, 3)

    # ---- Guidance (only when violated) ----
    # Guidance (only when violated)
    tips = []
    if hip_avg is not None:
        if hip_avg < HIP_MIN:
            tips.append("make the height of the hip higher")
        elif hip_avg > HIP_MAX:
            tips.append("make the height of the hip lower")

    if (dyL_n is not None and dyR_n is not None) and (dyL_n > KNEEHIP_MAX or dyR_n > KNEEHIP_MAX):
        tips.append("keep the thigh horizontal to the floor")

    # Only when we computed dx and knee is ahead of toe
    if (dxL is not None and dxL > KNEE_TOE_MAX) or (dxR is not None and dxR > KNEE_TOE_MAX):
        tips.append("do not let the knee exceed the tip of the toe")

    y0 = 76
    for g in tips[:3]:
        put_text(overlay, f"Tip: {g}", (12, y0), 0.85, TIP_COLOR, 2)
        y0 += 30

    # print to see log
    if debug:
        dbg = {
            "rot_deg": round(rot_deg, 3),
            "hip_avg": None if hip_avg is None else round(float(hip_avg), 3),
            "hip_L": None if hip_L is None else round(float(hip_L), 3),
            "hip_R": None if hip_R is None else round(float(hip_R), 3),
            "dyL_n": None if dyL_n is None else round(float(dyL_n), 4),
            "dyR_n": None if dyR_n is None else round(float(dyR_n), 4),
            "dxL": None if dxL is None else round(float(dxL), 4),
            "dxR": None if dxR is None else round(float(dxR), 4),
            "ok1": bool(ok1), "ok2": bool(ok2), "ok3": bool(ok3),
            "ok_all": bool(ok_all),
        }
        print("[DBG]", dbg)
        if not ok_all:
            failed = []
            if not ok1: failed.append(f"Cond1: hip angle not in [{HIP_MIN:.1f}, {HIP_MAX:.1f}]")
            if not ok2: failed.append(f"Cond2: |knee_y-hip_y|/body_h > {KNEEHIP_MAX:.2f}")
            if cond3_checks and not ok3:
                failed.append(f"Cond3: knee ahead of toe by > {KNEE_TOE_MAX:.2f}")
            elif not cond3_checks:
                failed.append("Cond3: not evaluated (missing knee/toe)")
            if failed:
                print("[DBG] Violations -> " + " | ".join(failed))

    # Optional CSV logging
    if csv_writer is not None:
        ts = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        for j, d in points.items():
            csv_writer.writerow([ts, 0, j, d['x'], d['y'], d['score']])

    return overlay


def run_on_webcam(infer, cam, width, height, save_csv=False, do_calib=True):
    csv_writer = None
    csv_file = None
    if save_csv:
        os.makedirs('media/output', exist_ok=True)
        csv_path = 'media/output/keypoints_log.csv'
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow(['ts_ms','person','joint','x','y','score'])

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        if csv_file: csv_file.close()
        raise RuntimeError("cannot open camera; check permissions or index")

    # Try to set resolution; verify
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    win_name = "Squat Checker – press 'q' to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            # Inference expects RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                out = next(infer(frame_rgb, return_vis=True, return_datasamples=True, show=False))
            except StopIteration:
                # Skip this frame if backend yields nothing
                continue

            overlay = evaluate_and_draw(frame_bgr, out, csv_writer, do_calibrate=do_calib)
            cv2.imshow(win_name, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()

def run_on_image(infer, img_path, save_path, save_csv=False, do_calib=True):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"cannot read image: {img_path}")

    # MMPose usually expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        out = next(infer(img_rgb, return_vis=True, return_datasamples=True, show=False))
    except StopIteration:
        raise RuntimeError("inferencer returned no results for the image")

    # optional CSV
    csv_writer = None
    csv_file = None
    if save_csv:
        os.makedirs('media/output', exist_ok=True)
        csv_path = 'media/output/keypoints_log.csv'
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow(['ts_ms','person','joint','x','y','score'])

    try:
        overlay = evaluate_and_draw(img_bgr, out, csv_writer, do_calibrate=do_calib)
    finally:
        if csv_file:
            csv_file.close()

    final_path = unique_save_path(save_path)
    cv2.imwrite(final_path, overlay)
    print(f"saved -> {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="store_true", help="Run on images in media/demo_input/")
    parser.add_argument("--webcam", action="store_true", help="Run live on webcam")
    parser.add_argument("--save_csv", action="store_true", help="Save keypoints to CSV")
    parser.add_argument("--no_calib", action="store_true", help="Disable calibration")
    args = parser.parse_args()

    infer = build_inferencer()

    if args.image:
        in_dir  = Path("media/demo_input")
        out_dir = Path("media/output")
        out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(glob.glob(str(in_dir / "*.[jp][pn]g")))  # jpg, jpeg, png
        if not image_paths:
            print(f"No images found in {in_dir}")
            return

        for img_path in image_paths:
            name = Path(img_path).stem
            save_path = out_dir / f"{name}_out.jpg"
            run_on_image(
                infer,
                img_path,
                save_path,
                save_csv=args.save_csv,
                do_calib=not args.no_calib,
            )

    elif args.webcam:
        run_on_webcam(
            infer,
            CAM_DEFAULT,
            FRAME_W,
            FRAME_H,
            save_csv=args.save_csv,
            do_calib=not args.no_calib,
        )
    else:
        print("Please specify either --image or --webcam")

if __name__ == "__main__":
    main()