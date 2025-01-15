import pytest
from fastapi.testclient import TestClient
from app.main import app
import base64
from PIL import Image
import io

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model" in response.json()

def test_generate_image():
    test_request = {
        "prompt": "test image",
        "style": "default",
        "steps": 30,
        "guidance_scale": 7.5
    }
    response = client.post("/generate", json=test_request)
    assert response.status_code == 200
    assert "image" in response.json()
    assert "metadata" in response.json()

def test_style_transfer():
    # Create a test image
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {
        'file': ('test.png', img_byte_arr, 'image/png')
    }
    response = client.post("/style-transfer", files=files, data={"style": "default"})
    assert response.status_code == 200
    assert "image" in response.json()

def test_invalid_style():
    test_request = {
        "prompt": "test image",
        "style": "invalid_style",
        "steps": 30,
        "guidance_scale": 7.5
    }
    response = client.post("/generate", json=test_request)
    assert response.status_code == 422