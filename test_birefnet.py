import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageSegmentation

class BirefNet:
    def __init__(self, model_id="ZhengPeng7/BiRefNet", base_size=1024):
        self.model_id = model_id
        self.base_size = base_size
        self.device = self._get_device()
        self.model = self._load_model()
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def _get_device(self):
        if torch.cuda.is_available():
            print("Hardware: Using NVIDIA GPU (CUDA)")
            return torch.device("cuda")
        print("Hardware: Using CPU")
        return torch.device("cpu")

    def _load_model(self):
        print(f"Loading model '{self.model_id}'...")
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            model.to(self.device)
            model.eval()
            print("Model loaded successfully.")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load the model. Error: {e}")

    def _preprocess(self, image_pil, target_size):
        # resize, return tensor
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            self.normalize
        ])
        return transform(image_pil).unsqueeze(0).to(self.device)

    def _infer_single(self, input_tensor):
        # inference single tensor
        model_dtype = next(self.model.parameters()).dtype
        input_tensor = input_tensor.to(dtype=model_dtype)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            if isinstance(outputs, (list, tuple)):
                pred = outputs[-1]
            elif hasattr(outputs, 'logits'):
                pred = outputs.logits
            else:
                pred = outputs
            return pred.sigmoid()

    def _infer_tta(self, input_tensor):
        # Test-Time Augmentation (Horizontal Flip)
        # inference original
        pred_normal = self._infer_single(input_tensor)
        
        # inference - horizontal flip 
        input_flipped = torch.flip(input_tensor, dims=[3])
        pred_flipped = self._infer_single(input_flipped)
        pred_flipped_restored = torch.flip(pred_flipped, dims=[3])
        
        # avg
        pred_ensemble = (pred_normal + pred_flipped_restored) / 2.0
        return pred_ensemble[0].squeeze().cpu().float().numpy()

    def _get_edge_weight_map(self, image_np):
        # Edge-aware weight
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
        edges = cv2.Canny(gray, 50, 150)
        
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        edges = cv2.GaussianBlur(edges, (15, 15), 0)
        return edges.astype(np.float32) / 255.0

    def _guided_filter(self, guide_img, mask, radius=20, eps=1e-5):
        # refine edge
        I = cv2.cvtColor(guide_img, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        p = mask.astype(np.float64)

        # box filter
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

        q = mean_a * I + mean_b
        return np.clip(q, 0, 1).astype(np.float32)

    def _remove_small_noise(self, mask_np, threshold=0.5, min_size_ratio=0.05):
        binary = (mask_np > threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels <= 1:
            return mask_np

        max_area = np.max(stats[1:, cv2.CC_STAT_AREA])
        
        valid_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= max_area * min_size_ratio]
        
        clean_mask = mask_np.copy()

        mask_to_zero = np.isin(labels, valid_labels, invert=True) & (labels != 0)
        clean_mask[mask_to_zero] = 0.0
        
        return clean_mask

    def process_image(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' not found.")
            return

        try:
            print(f"\nProcessing '{input_path}'...")
            original_pil = Image.open(input_path).convert("RGB")
            orig_w, orig_h = original_pil.size
            orig_np = np.array(original_pil)

            # ---------------------------------------------------------
            # 1. Multi-scale Inference & TTA
            # ---------------------------------------------------------
            scales = [0.75, 1.0, 1.25]
            predictions = {}
            
            for scale in scales:
                target_size = int(self.base_size * scale)
                print(f" - Running inference at scale {scale}x ({target_size}px) with TTA...")
                input_tensor = self._preprocess(original_pil, target_size)
                pred_np = self._infer_tta(input_tensor)
                
                pred_resized = cv2.resize(pred_np, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
                predictions[scale] = pred_resized

            # ---------------------------------------------------------
            # 2. Edge-aware Weighting
            # ---------------------------------------------------------
            print(" - Combining predictions using edge-aware weighting...")
            edge_weight = self._get_edge_weight_map(orig_np)
            
            mask_flat = (predictions[0.75] + predictions[1.0]) / 2.0
            
            combined_mask = edge_weight * predictions[1.25] + (1.0 - edge_weight) * mask_flat

            # ---------------------------------------------------------
            # 3. Post-processing (Noise Removal & Refinement)
            # ---------------------------------------------------------
            print(" - Cleaning up noise and refining hair/edges (Guided Filter)...")
            cleaned_mask = self._remove_small_noise(combined_mask)
            
            refined_mask = self._guided_filter(orig_np, cleaned_mask, radius=10, eps=1e-4)
            
            refined_mask = np.clip((refined_mask - 0.05) / 0.9, 0, 1)

            # ---------------------------------------------------------
            # 4. Final Export
            # ---------------------------------------------------------
            final_mask_pil = Image.fromarray((refined_mask * 255).astype(np.uint8))
            
            result_image = original_pil.copy()
            result_image.putalpha(final_mask_pil)
            result_image.save(output_path, format="PNG")
            
            print(f"Success! High-resolution transparent image saved to '{output_path}'")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example Usage
    INPUT_IMAGE_PATH = "image/sample1.jpg"
    OUTPUT_IMAGE_PATH = "sample_output.png" 
    
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"[{INPUT_IMAGE_PATH}] file not found. Creating a dummy image...")
        dummy_img = Image.new('RGB', (1024, 1024), color = 'red')
        dummy_img.save(INPUT_IMAGE_PATH)

    remover = BirefNet(base_size=1024)
    remover.process_image(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)