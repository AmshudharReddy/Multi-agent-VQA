import torch
from torch.cuda.amp import autocast
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "tools/clip/CLIP-Count"))

from util import misc  # Adjust import path to your repo
from run import Model

SCALE_FACTOR = 60

class ClipCountAgent:
    def __init__(self, ckpt_path, device="cpu"):
        """
        Initializes the ClipCountAgent.
        Args:
            ckpt_path: Path to CLIP-Count checkpoint (.ckpt file)
            device: "cpu" or "cuda"
        """
        self.model = Model.load_from_checkpoint(ckpt_path, strict=False)
        self.model.eval()
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(device)

    def detect_count(self, image_np, needed_object):
        """
        Runs CLIP-Count inference.
        Args:
            image_np: NumPy image in BGR format
            needed_object: str prompt describing the object to count
        Returns:
            count (int), heatmap_overlay (PIL.Image)
        """
        # Convert BGR to RGB and to tensor
        img = torch.from_numpy(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img = img.float() / 255.
        img = torch.clamp(img, 0, 1)

        prompt = [needed_object]

        with torch.no_grad():
            with autocast():
                # Save original dimensions
                raw_h, raw_w = img.shape[2:]
                
                # Resize height to 384, keep aspect ratio
                scale_factor = 384 / raw_h
                new_w = int(raw_w * scale_factor)
                img = TF.resize(img, (384, new_w))

                # Sliding window
                patches, _ = misc.sliding_window(img, stride=128)
                patches = torch.from_numpy(patches).float().to(self.device)

                # Repeat prompts for all patches
                prompt_batch = np.repeat(prompt, patches.shape[0], axis=0)

                # Forward pass
                output = self.model.forward(patches, prompt_batch)
                output.unsqueeze_(1)

                # Composite patches back into full image
                output = misc.window_composite(output, stride=128)
                output = output.squeeze(1)

                # Crop to original resized width
                output = output[:, :, :new_w]

            # Predicted count
            pred_cnt = torch.sum(output[0] / SCALE_FACTOR).item()
            count = int(round(pred_cnt))

            # Create heatmap overlay
            pred_density = output[0].detach().cpu().numpy()
            pred_density = pred_density / pred_density.max()
            pred_density_write = 1. - pred_density
            pred_density_write = cv2.applyColorMap(np.uint8(255 * pred_density_write), cv2.COLORMAP_JET) / 255.

            img_np = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            heatmap_pred = 0.33 * img_np + 0.67 * pred_density_write
            heatmap_pred = heatmap_pred / heatmap_pred.max()
            heatmap_pred = (heatmap_pred * 255).astype(np.uint8)

        # return count, Image.fromarray(cv2.cvtColor(heatmap_pred, cv2.COLOR_BGR2RGB))
        return f"[estimated count] {count}", Image.fromarray(cv2.cvtColor(heatmap_pred, cv2.COLOR_BGR2RGB))

