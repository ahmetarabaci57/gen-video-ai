from pydantic import BaseModel, Field


class GenerateReelsRequest(BaseModel):
    idea: str = Field(..., min_length=3, max_length=2000, description="Turkish reel idea")
    num_frames: int = Field(default=5, ge=1, le=12)
    seconds_per_frame: float = Field(default=2.2, ge=0.8, le=8.0)


class GenerateReelsResponse(BaseModel):
    status: str
    idea: str
    english_prompts: list[str]
    frames: list[str]
    video_path: str
    message: str
