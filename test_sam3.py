import torch 

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else: device = "cpu"
print(f"using device: {device}")

model = build_sam3_image_model().to(device)
processor = Sam3Processor(model,confidence_threshold=0.7)

image_path = f"image/sample{i}.jpg"
image = Image.open(image_path).convert("RGB")
inference_state = processor.set_image(image)

text_prompt = "hat"
output = processor.set_text_prompt(
    state=inference_state,
    prompt=text_prompt
)

img0 = Image.open(image_path)
plot_results(img0, inference_state)
plt.savefig("result_output.jpg")

masks = output.get('masks') 

if masks is not None and len(masks) > 0:
    mask_np = masks[0].cpu().numpy().squeeze() 
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_uint8, mode='L')
    mask_image.save(f"mask{i}.png")
    print("done.")
else:
    print("error.")