from pydantic import BaseModel, Field

class GenerateReelsRequest(BaseModel):
    idea: str = Field(..., min_length=3, max_length=2000, description="Turkish reel idea")

class GenerateReelsResponse(BaseModel):
    status: str
    idea: str
    english_prompts: list[str]
    frames: list[str]
    video_path: str
    message: str
