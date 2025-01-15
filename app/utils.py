import torch
import base64
from PIL import Image
import io

def encode_image(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def decode_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))