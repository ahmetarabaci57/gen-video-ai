from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from config import settings
from schemas import GenerateReelsRequest, GenerateReelsResponse
from services.comfy_service import ComfyService, ComfyServiceError
from services.llm_service import LLMService, LLMServiceError
from services.video_service import VideoService, VideoServiceError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("reels")

app = FastAPI(
    title="Local Instagram Reels Generator",
    version="1.0.0",
    description="Turkish idea -> Ollama -> ComfyUI (GPU) -> MoviePy (CPU).",
)

llm_service = LLMService()
comfy_service = ComfyService()
video_service = VideoService()
gpu_lock = asyncio.Lock()
INDEX_HTML = (settings.output_dir.parent / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate-reels", response_model=GenerateReelsResponse)
async def generate_reels(payload: GenerateReelsRequest) -> GenerateReelsResponse:
    if gpu_lock.locked():
        raise HTTPException(
            status_code=429,
            detail="Another reel is using the GPU. Wait until it finishes to protect 4GB VRAM.",
        )

    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    job_dir = settings.output_dir / job_id

    async with gpu_lock:
        try:
            logger.info("Step 1/3 Ollama translation | job=%s", job_id)
            prompts = await llm_service.idea_to_image_prompts(payload.idea, payload.num_frames)

            logger.info("Step 2/3 ComfyUI generation | job=%s frames=%s", job_id, len(prompts))
            frames = await comfy_service.generate_frames(prompts, job_dir / "frames")

            logger.info("Step 3/3 MoviePy CPU render | job=%s", job_id)
            video_path = await asyncio.to_thread(
                video_service.render_reel,
                frames,
                job_dir / "reel.mp4",
                payload.seconds_per_frame,
            )
        except LLMServiceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ComfyServiceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except VideoServiceError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected pipeline failure")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    return GenerateReelsResponse(
        status="completed",
        idea=payload.idea,
        english_prompts=prompts,
        frames=[str(path) for path in frames],
        video_path=str(video_path),
        message="Reel generated locally. GPU was used only for ComfyUI; video render ran on CPU.",
    )


@app.get("/api/download/{job_id}")
async def download_reel(job_id: str) -> FileResponse:
    video_path = settings.output_dir / job_id / "reel.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(path=video_path, media_type="video/mp4", filename=f"{job_id}.mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
