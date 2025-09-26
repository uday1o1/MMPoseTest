# webcam_pose.py
import cv2
import torch
from mmpose.apis import MMPoseInferencer

device = "cpu"
print("Device:", device)

working_pose2d='td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
working_pose2d_weights='td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
working_det_model='rtmdet_tiny_8xb32-300e_coco'

POSE = 'rtmpose-l_8xb64-270e_coco-wholebody-256x192.py'
POSE_WTS = 'rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth'
DET  = 'rtmdet_nano_320-8xb32_coco-person.py'

inferencer = MMPoseInferencer(
    pose2d=POSE,
    pose2d_weights=POSE_WTS,
    det_model=DET,
    device=device
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera 0. Check macOS Camera permissions; try index 1/2.")

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    out = next(inferencer(frame_bgr, return_vis=True, show=False))
    vis_rgb = out["visualization"][0]
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("MMPose (press q to quit)", vis_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
