import os, numpy as np, platform, torch
from PIL import Image
from mmcv import imread
from mmengine.registry import VISUALIZERS
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples
from mmpose.utils import register_all_modules

register_all_modules()

CFG = 'models/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
CKPT = 'models/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
IMG  = 'media/demo.jpg'

device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
print("Arch:", platform.machine(), "| Torch:", torch.__version__, "| Device:", device)

# 1) init model
model = init_model(CFG, CKPT, device=device)

# 2) bbox covering whole image (good for single-person photos)
w, h = Image.open(IMG).size
bboxes = np.array([[0, 0, w - 1, h - 1]], dtype=float)

# 3) inference
result_generator = inference_topdown(model, IMG, bboxes=bboxes)
print(f"Instances detected: {len(result_generator)}")
if result_generator:
    kpts = result_generator[0].pred_instances.keypoints  # (N, K, 2)
    print("Keypoints shape:", kpts.shape)
    # print('results file: ', result_generator)

# 4) visualize
os.makedirs('outputs', exist_ok=True)
img = imread(IMG)  # BGR ndarray
vis = VISUALIZERS.build(dict(type='PoseLocalVisualizer', name='pose_vis',
                             vis_backends=[dict(type='LocalVisBackend')]))
vis.set_dataset_meta(model.dataset_meta)
vis.add_datasample(
    name='result',
    image=img,
    data_sample=merge_data_samples(result_generator),
    draw_gt=False, draw_heatmap=False, draw_bbox=True,
    show=False, out_file='outputs/pose_vis.jpg'
)
print("Saved -> outputs/pose_vis.jpg")