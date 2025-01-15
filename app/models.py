from pydantic import BaseModel, Field

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    style: str = Field(default="default", pattern="^(default|anime|oil_painting)$")
    steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)