# AI Image Generation Backend

A FastAPI backend service for AI-powered image generation using Stable Diffusion.

## 📁 Project Structure
```
image-gen-backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Features
- Text-to-image generation using Stable Diffusion
- Style transfer capabilities
- Multiple style options (anime, oil painting, etc.)
- RESTful API endpoints
- CORS support
- Error handling and validation

## 🛠️ Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Docker (optional)

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-gen-backend.git
cd image-gen-backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Server

### Development
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker
```bash
docker build -t image-gen-backend .
docker run -p 8000:8000 image-gen-backend
```

## 📚 API Documentation

### Generate Image
POST `/generate`

Request body:
```json
{
    "prompt": "string",
    "style": "string",
    "steps": "integer",
    "guidance_scale": "float"
}
```

Response:
```json
{
    "image": "base64_encoded_string"
}
```

### Style Transfer
POST `/style-transfer`

Form data:
- file: image file
- style: string

Response:
```json
{
    "image": "base64_encoded_string"
}
```

## 🔧 Configuration
Environment variables:
- `MODEL_ID`: Stable Diffusion model ID (default: "runwayml/stable-diffusion-v1-5")
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `MAX_STEPS`: Maximum inference steps (default: 50)

## 🧪 Testing
```bash
pytest tests/
```

## 📝 License
MIT License

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support
For support, email your-email@example.com or create an issue in the repository.
