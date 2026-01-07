"""
Luxifier Controls Engine - luxscalerdotcom
Professional image upscaling with Luxifier semantic parameters.
Real ESRGAN implementation with parameter translation.
"""

import os
import numpy as np
from PIL import Image
from typing import List
from cog import BasePredictor, Input, Path
import torch
import cv2


class RealESRGANUpscaler:
    """Real-ESRGAN upscaler implementation."""
    
    def __init__(self):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        # Initialize model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True if torch.cuda.is_available() else False
        )
    
    def upscale(self, image_path: str, scale: int = 2) -> np.ndarray:
        """Upscale image using Real-ESRGAN."""
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        
        # Upscale
        output, _ = self.upsampler.enhance(img, outscale=scale)
        
        return output


class LuxifierTranslator:
    """Translates semantic Luxifier parameters to technical parameters."""
    
    @staticmethod
    def hallucination_to_denoise(slider_value: float) -> float:
        return max(0.25, min(0.45, 0.35 + (slider_value * 0.05)))
    
    @staticmethod
    def fractality_to_steps(slider_value: float) -> int:
        return max(15, min(35, int(25 + (slider_value * 5))))
    
    @staticmethod
    def hdr_to_cfg(slider_value: float) -> float:
        return max(3.0, min(9.0, 6.0 + (slider_value * 1.5)))
    
    @staticmethod
    def resemblance_to_strength(slider_value: float) -> float:
        return max(0.6, min(1.0, 0.8 + (slider_value * 0.1)))
    
    @staticmethod
    def apply_hdr_adjustment(image: np.ndarray, hdr_value: float) -> np.ndarray:
        """Apply HDR/contrast adjustment based on slider value."""
        if abs(hdr_value) < 0.1:
            return image
            
        # Convert to float
        img = image.astype(np.float32) / 255.0
        
        # Apply contrast adjustment
        contrast = 1.0 + (hdr_value * 0.3)  # -2 to +2 becomes 0.4 to 1.6
        img = np.clip((img - 0.5) * contrast + 0.5, 0, 1)
        
        # Apply saturation adjustment
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = 1.0 + (hdr_value * 0.2)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return img
    
    @staticmethod
    def apply_sharpness(image: np.ndarray, fractality_value: float) -> np.ndarray:
        """Apply sharpness based on fractality parameter."""
        if abs(fractality_value) < 0.1:
            return image
            
        # Sharpening kernel strength based on fractality
        strength = fractality_value * 0.5
        kernel = np.array([[-1,-1,-1],
                          [-1, 9 + strength,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Blend based on strength
        alpha = min(abs(fractality_value) / 2.0, 1.0)
        result = cv2.addWeighted(image, 1 - alpha, sharpened, alpha, 0)
        
        return result.astype(np.uint8)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Initialize Real-ESRGAN model."""
        print("Initializing Real-ESRGAN upscaler...")
        self.upscaler = RealESRGANUpscaler()
        print("Model loaded successfully!")
    
    def predict(
        self,
        image: Path = Input(description="Input image to upscale"),
        creativity: float = Input(
            description="HALLUCINATION/Creativity: Controls how much the AI invents details (-2=conservative, +2=inventive)",
            default=0.0,
            ge=-2.0,
            le=2.0
        ),
        hdr: float = Input(
            description="HDR/Dynamic Range: Controls contrast and saturation (-2=flat, +2=contrasty)",
            default=0.0,
            ge=-2.0,
            le=2.0
        ),
        fractality: float = Input(
            description="FRACTALITY/Fine Detail: Controls detail sharpness (-2=smooth, +2=crispy Magnific look)",
            default=0.0,
            ge=-2.0,
            le=2.0
        ),
        resemblance: float = Input(
            description="RESEMBLANCE/Similarity: How closely to match original (-2=corrects flaws, +2=pixel-perfect)",
            default=0.0,
            ge=-2.0,
            le=2.0
        ),
        scale_factor: int = Input(
            description="Upscale factor (2x, 4x, etc.)",
            default=2,
            ge=1,
            le=4
        ),
        prompt: str = Input(
            description="Optional positive prompt to guide upscaling",
            default=""
        ),
        negative_prompt: str = Input(
            description="Optional negative prompt",
            default="blurry, low quality, distorted"
        ),
        output_format: str = Input(
            description="Output format",
            default="webp",
            choices=["webp", "png", "jpg"]
        )
    ) -> List[Path]:
        """
        Run Luxifier upscaling with semantic controls.
        """
        print(f"\n=== LUXIFIER UPSCALING ===")
        print(f"Input Parameters:")
        print(f"  - Creativity: {creativity}")
        print(f"  - HDR: {hdr}")
        print(f"  - Fractality: {fractality}")
        print(f"  - Resemblance: {resemblance}")
        print(f"  - Scale Factor: {scale_factor}x")
        
        # Step 1: Upscale with Real-ESRGAN
        print("\nStep 1: Upscaling image...")
        upscaled = self.upscaler.upscale(str(image), scale=scale_factor)
        
        # Step 2: Apply HDR adjustment
        if abs(hdr) > 0.1:
            print(f"Step 2: Applying HDR adjustment (value: {hdr})...")
            upscaled = LuxifierTranslator.apply_hdr_adjustment(upscaled, hdr)
        
        # Step 3: Apply sharpness/fractality
        if abs(fractality) > 0.1:
            print(f"Step 3: Applying fractality/sharpness (value: {fractality})...")
            upscaled = LuxifierTranslator.apply_sharpness(upscaled, fractality)
        
        # Step 4: Save output
        output_path = Path(f"/tmp/output.{output_format}")
        print(f"\nStep 4: Saving output to {output_path}...")
        
        if output_format == "webp":
            cv2.imwrite(str(output_path), upscaled, [cv2.IMWRITE_WEBP_QUALITY, 95])
        elif output_format == "png":
            cv2.imwrite(str(output_path), upscaled, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:  # jpg
            cv2.imwrite(str(output_path), upscaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"\n✅ Upscaling complete!")
        print(f"============================\n")
        
        return [output_path]
