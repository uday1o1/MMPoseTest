import cv2
import torch
from mmpose.apis import MMPoseInferencer

device = "cpu"
print("Device:", device)

POSE = 'models/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py'
POSE_WTS = 'models/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth'
DET  = 'models/rtmdet_nano_320-8xb32_coco-person.py'

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
