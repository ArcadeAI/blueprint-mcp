import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from prompts import AspectRatio, ImageSize


class GenerationConfig(BaseModel):
    aspect_ratio: AspectRatio = Field(default=AspectRatio.LANDSCAPE)
    image_size: ImageSize = Field(default=ImageSize.HIGH)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class GenerationResult(BaseModel):
    success: bool
    file_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    error: Optional[str] = None
    model_used: str = "gemini-3-pro-image-preview"


MODELS = {
    "pro": "gemini-3-pro-image-preview",
    "flash": "gemini-3.1-flash-image-preview",
}


class GeminiImageGenerator:
    MAX_OUTPUT_TOKENS = 32768

    def __init__(self, api_key: str, output_dir: Optional[Path] = None, model: str = "pro"):
        self.api_key = api_key
        self.model_name = MODELS.get(model, MODELS["pro"])
        self.output_dir = output_dir or Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str,
        config: GenerationConfig = GenerationConfig(),
        filename_prefix: str = "diagram"
    ) -> GenerationResult:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )
        ]
        return self.generate_with_contents(contents, config, filename_prefix)

    def generate_with_contents(
        self,
        contents: list,
        config: GenerationConfig = GenerationConfig(),
        filename_prefix: str = "diagram"
    ) -> GenerationResult:
        try:
            generation_config = types.GenerateContentConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                max_output_tokens=self.MAX_OUTPUT_TOKENS,
                response_modalities=["IMAGE"],
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
                ],
                image_config=types.ImageConfig(
                    aspect_ratio=config.aspect_ratio.value,
                    image_size=config.image_size.value,
                ),
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generation_config,
            )

            image_data = self._extract_image_data(response)
            if not image_data:
                return GenerationResult(success=False, error="No image data in response")

            file_path, width, height = self._save_image(image_data, filename_prefix)

            return GenerationResult(
                success=True,
                file_path=str(file_path),
                width=width,
                height=height,
                model_used=self.model_name
            )

        except Exception as e:
            return GenerationResult(success=False, error=self._format_error(e))
    
    def _extract_image_data(self, response) -> Optional[bytes]:
        if not response.candidates:
            return None
            
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                img_data = part.inline_data.data
                if isinstance(img_data, str):
                    return base64.b64decode(img_data)
                else:
                    return img_data
        
        return None
    
    def _save_image(self, image_data: bytes, prefix: str) -> tuple[Path, int, int]:
        image = Image.open(BytesIO(image_data))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        file_path = self.output_dir / filename
        image.save(file_path, "PNG")
        return file_path, image.width, image.height
    
    def _format_error(self, error: Exception) -> str:
        error_str = str(error)
        
        if "quota" in error_str.lower():
            return f"API quota exceeded: {error_str}"
        elif "401" in error_str or "authentication" in error_str.lower():
            return f"Authentication failed: {error_str}"
        elif "404" in error_str or "not found" in error_str.lower():
            return f"Model not found: {error_str}"
        elif "billing" in error_str.lower():
            return f"Billing required: {error_str}"
        else:
            return f"Generation failed: {error_str}"


NanoBananaPro = GeminiImageGenerator  # backwards compat


class DiagramGenerator:
    def __init__(self, api_key: str, output_dir: Optional[Path] = None, model: str = "pro"):
        self.client = GeminiImageGenerator(api_key=api_key, output_dir=output_dir, model=model)
    
    def generate_from_prompt(
        self,
        prompt: str,
        aspect_ratio: str = "16:9",
        resolution: str = "2K",
        filename_prefix: str = "diagram"
    ) -> GenerationResult:
        config = GenerationConfig(
            aspect_ratio=AspectRatio(aspect_ratio),
            image_size=ImageSize(resolution)
        )
        
        return self.client.generate(
            prompt=prompt,
            config=config,
            filename_prefix=filename_prefix
        )
