from __future__ import annotations

import asyncio
import logging
import uuid
import re
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from config import settings
from schemas import GenerateReelsRequest, GenerateReelsResponse
from services.comfy_service import ComfyService, ComfyServiceError
from services.llm_service import LLMService, LLMServiceError
from services.video_service import VideoService, VideoServiceError
from services.audio_service import AudioService 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("reels")

# 🚨 FastAPI Başlatma Alanı
app = FastAPI(
    title="Local Instagram Reels Generator",
    version="1.0.0",
    description="Turkish Text -> CPU TTS -> Ollama -> ComfyUI (GPU) -> MoviePy (CPU).",
)

llm_service = LLMService()
comfy_service = ComfyService()
video_service = VideoService()
audio_service = AudioService()  
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
            detail="GPU şu an başka bir video üretiyor, lütfen bekleyin.",
        )

    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    job_dir = settings.output_dir / job_id

    async with gpu_lock:
        try:
            speech_text = payload.idea  # Default olarak her şeyi oku (eğer tırnak yoksa)
            
            # Çift tırnak (") veya tek tırnak (') içerisindeki metni bulur
            quotes = re.findall(r'["\']([^"\']+)["\']', payload.idea)
            
            if quotes:
                speech_text = quotes[0]  # Sadece ilk tırnak içindeki cümleyi seslendir
                logger.info("Filtrelenmiş Karakter Repliği (Seslendirilecek): %s", speech_text)
            else:
                logger.info("Tırnak bulunamadı, tüm metin seslendiriliyor.")

            logger.info("Adım 1/4: CPU üzerinde Sadece Repliğin Ses Üretimi | job=%s", job_id)
            audio_path = job_dir / "speech.mp3"
            # payload.idea yerine filtrelenmiş speech_text'i gönderiyoruz!
            await asyncio.to_thread(audio_service.generate_turkish_audio, speech_text, audio_path)

            logger.info("Adım 2/4: Ollama ile Prompt Çevirisi ve Temizliği | job=%s", job_id)
            # Görsel üretimi için Ollama'ya metnin tamamını gönderiyoruz ki sahneyi anlayabilsin
            prompts = await llm_service.idea_to_image_prompts(payload.idea, 1) 
            if not prompts:
                raise LLMServiceError("Ollama herhangi bir prompt üretemedi.")
            
            if isinstance(prompts, list):
                raw_prompt = prompts[0] if len(prompts) > 0 else str(prompts)
            else:
                raw_prompt = str(prompts)
                
            clean_prompt = raw_prompt.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
            anime_prompt = f"anime style illustration, masterpiece, 8k, talking avatar portrait, {clean_prompt}"

            logger.info("Adım 3/4: ComfyUI ile Tek Kare Karakter Çizimi | job=%s", job_id)
            frames = await comfy_service.generate_frames([anime_prompt], job_dir / "frames")

            logger.info("Adım 4/4: MoviePy ile Ses ve Görüntü Birleştirme | job=%s", job_id)
            video_path = await asyncio.to_thread(
                video_service.render_talking_reel,
                frames,
                audio_path,
                job_dir / "reel.mp4"
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
        english_prompts=[anime_prompt],
        frames=[str(path) for path in frames],
        video_path=str(video_path),
        message="Ses üretildi ve tek kare görsel ile başarıyla birleştirildi!",
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
