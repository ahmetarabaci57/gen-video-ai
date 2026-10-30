import logging
import os
from pathlib import Path
from moviepy import ImageClip, AudioFileClip

logger = logging.getLogger(__name__)

class VideoServiceError(RuntimeError):
    pass

class VideoService:
    """CPU-only single image and audio fusion module."""

    def render_talking_reel(self, frames: list[Path] | Path, audio_path: Path, output_path: Path) -> Path:
        """Tek bir görseli alır, ses süresi kadar uzatır ve sesi üzerine giydirir."""
        if not frames:
            raise VideoServiceError("Görüntü oluşturulamadığı için video kurgulanamıyor.")
            
        # 🚨 EĞER LİSTE OLARAK GELDİYSE İLK ELEMANI AYIKLA
        frame_path = frames[0] if isinstance(frames, list) else frames

        
        if not frame_path.exists():
            raise VideoServiceError(f"Görsel dosyası bulunamadı: {frame_path}")
        if not audio_path.exists():
            raise VideoServiceError(f"Ses dosyası bulunamadı: {audio_path}")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sesi yükle ve tam süresini hesapla
            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration
            
            logger.info("Loading single frame for video elongation. Duration: %s sec", duration)
            
            # Resmi dikey boyutta ses süresi kadar uzatarak video klibi yap
            video_clip = (
                ImageClip(str(frame_path))
                .resized((1080, 1920))  # Dikey Reels Standart Çözünürlüğü
                .with_duration(duration)
                .with_fps(24)
            )
            
            # Ses kanalını videoya ekle
            video_clip = video_clip.with_audio(audio_clip)
            
            logger.info("Rendering final synced MP4 video (CPU only) to %s", output_path)
            video_clip.write_videofile(
                str(output_path),
                fps=24,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
                threads=os.cpu_count() or 4,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                logger=None,
            )
            
            # Kaynakları serbest bırak (Memory Leak önlemek için kritik)
            audio_clip.close()
            video_clip.close()
            
            return output_path
        except Exception as exc:
            raise VideoServiceError(f"MoviePy video senkronizasyonu başarısız: {exc}") from exc
