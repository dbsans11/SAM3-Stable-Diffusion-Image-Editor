import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageFilter
import numpy as np
import cv2
import os

# ==========================================
# --- Configuration & SDXL Model Initialization ---
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
use_vram_optimization = True  
model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

print(f"[{device.upper()}] Initializing device and SDXL pipeline...")
try:
    pipeline = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None
    )
    pipeline = pipeline.to(device)

    if use_vram_optimization and device == "cuda":
        pipeline.enable_model_cpu_offload()
        print("VRAM optimization (model CPU offload) enabled.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ==========================================
# --- [Core] Patch-Based High-Resolution SDXL Inpainting Pipeline ---
# ==========================================
def run_inpainting_pipeline(
    pipe, 
    prompt: str, 
    negative_prompt: str, 
    init_image: Image.Image,
    mask_image: Image.Image, 
    max_size: int = 1024,      # Target resolution for the cropped patch (1024 highly recommended for SDXL)
    up_shift: int = 0,         # Vertical mask shift (set to 0 for standard object removal)
    dilation_kernel: int = 30, # Kernel size for mask dilation
    blur_radius: int = 6,      # Gaussian blur radius for seamless boundary blending (feathering)
    crop_margin: int = 256     # Contextual padding to include surrounding textures (e.g., skin context)
):
    print("------------------------------------------")
    print("Starting HQ Patch-based SDXL inpainting (Zero Resolution Loss)...")
    
    orig_w, orig_h = init_image.size

    # ------------------------------------------
    # 1. Mask Preprocessing (Dilation & Translation)
    # ------------------------------------------
    mask_np = np.array(mask_image.convert("L"))
    h, w = mask_np.shape
    
    m_matrix = np.float32([[1, 0, 0], [0, 1, -up_shift]])
    shifted_mask = cv2.warpAffine(mask_np, m_matrix, (w, h))
    combined_mask = cv2.bitwise_or(mask_np, shifted_mask)
    
    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    final_mask_np = cv2.dilate(combined_mask, kernel, iterations=1)
    
    # ------------------------------------------
    # 2. Bounding Box Calculation & ROI Cropping
    # ------------------------------------------
    # Extract coordinates of the masked region
    coords = cv2.findNonZero(final_mask_np)
    if coords is None:
        print("Warning: Empty mask detected. Returning original image.")
        return init_image
        
    x, y, box_w, box_h = cv2.boundingRect(coords)
    
    # Apply margin to provide sufficient context for the model
    crop_x1 = max(0, x - crop_margin)
    crop_y1 = max(0, y - crop_margin)
    crop_x2 = min(orig_w, x + box_w + crop_margin)
    crop_y2 = min(orig_h, y + box_h + crop_margin)
    
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    
    # Extract the ROI (Region of Interest) from the original high-res image
    crop_img = init_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    expanded_mask_image = Image.fromarray(final_mask_np)
    crop_mask = expanded_mask_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
    print(f"ROI defined: {crop_w}x{crop_h} at coordinates ({crop_x1}, {crop_y1})")

    # ------------------------------------------
    # 3. Resize ROI to SDXL Specifications
    # ------------------------------------------
    # Resize only the extracted patch to standard SDXL dimensions to preserve overall image fidelity
    ratio = max_size / max(crop_w, crop_h)
    new_w = int((crop_w * ratio) // 8 * 8)
    new_h = int((crop_h * ratio) // 8 * 8)
    
    resized_crop_img = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_crop_mask = crop_mask.resize((new_w, new_h), Image.Resampling.NEAREST)
    resized_crop_mask = resized_crop_mask.point(lambda p: 255 if p > 128 else 0)

    # ------------------------------------------
    # 4. SDXL Inference (Processed strictly within the ROI)
    # ------------------------------------------
    print("Running SDXL inference on the cropped ROI...")
    with torch.no_grad():
        inpainted_crop = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=resized_crop_img,
            mask_image=resized_crop_mask,
            guidance_scale=8,        # Elevated scale to strictly enforce skin texture generation
            num_inference_steps=85,
        ).images[0]

    # ------------------------------------------
    # 5. Restoration & Seamless Compositing
    # ------------------------------------------
    print("Compositing the generated patch back into the original image...")
    # Upscale/Downscale the inpainted patch back to its original cropped dimensions
    restored_crop = inpainted_crop.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
    
    # Generate a feathered mask for seamless alpha blending
    crop_mask_blurred = crop_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Paste the restored patch onto the exact coordinates of the original image
    final_result = init_image.copy()
    final_result.paste(restored_crop, (crop_x1, crop_y1), crop_mask_blurred)
    
    print("Inpainting pipeline executed successfully.")
    print("------------------------------------------")
    return final_result

# ==========================================
# --- Execution / Main ---
# ==========================================
if __name__ == "__main__":
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    image_path = os.path.join(test_dir, "origin.jpg")
    mask_path = os.path.join(test_dir, "mask.png")
    output_path = os.path.join(test_dir, "result_sdxl.png")

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"Error: Missing input files. Please ensure 'origin.jpg' and 'mask.png' exist in the '{test_dir}' directory.")
        exit()

    origin_image = Image.open(image_path).convert("RGB")
    origin_mask = Image.open(mask_path).convert("L")

    positive_prompt = "an arm without a watch, an arm without a smart watch, an arm without an armband, Natural-looking skin, arm skin"
    negative_prompt = "watch, smart watch, clock, armband, wristband, bracelet, accessories, an arm with a watch, an arm with a smart watch"

    final_image = run_inpainting_pipeline(
        pipe=pipeline,
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        init_image=origin_image,
        mask_image=origin_mask,
        max_size=1024,      
        up_shift=0,         
        dilation_kernel=50,
        blur_radius=15,    
        crop_margin=512     
    )

    final_image.save(output_path)
    print(f"Process complete. Output saved to '{output_path}'.")