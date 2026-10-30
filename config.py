from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = 180.0

    comfy_http_url: str = "http://127.0.0.1:8188"
    comfy_ws_url: str = "ws://127.0.0.1:8188/ws"
    comfy_timeout_seconds: float = 300.0
    workflow_path: Path = ROOT_DIR / "workflow_api.json"

    output_dir: Path = ROOT_DIR / "output"
    video_fps: int = 30
    video_width: int = 1080
    video_height: int = 1920
    default_frames: int = 5
    default_seconds_per_frame: float = 2.2
    crossfade_seconds: float = 0.45


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
