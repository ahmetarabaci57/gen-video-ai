from __future__ import annotations

import logging
import os
from pathlib import Path

from moviepy import ImageClip, concatenate_videoclips, vfx

from config import settings

logger = logging.getLogger(__name__)


class VideoServiceError(RuntimeError):
    pass


class VideoService:
    """CPU-only assembly. Do not import torch or call CUDA from this module."""

    def render_reel(
        self,
        frames: list[Path],
        output_path: Path,
        seconds_per_frame: float,
    ) -> Path:
        if not frames:
            raise VideoServiceError("No frames were provided for video rendering.")

        missing = [str(path) for path in frames if not path.exists()]
        if missing:
            raise VideoServiceError(f"Missing frame files: {missing}")

        clips: list[ImageClip] = []
        try:
            fade = min(settings.crossfade_seconds, max(0.05, seconds_per_frame / 3))
            for path in frames:
                clip = (
                    ImageClip(str(path))
                    .resized((settings.video_width, settings.video_height))
                    .with_duration(seconds_per_frame)
                    .with_fps(settings.video_fps)
                )
                if len(frames) > 1 and fade > 0:
                    clip = clip.with_effects([vfx.CrossFadeIn(fade), vfx.CrossFadeOut(fade)])
                clips.append(clip)

            padding = -fade if len(clips) > 1 else 0
            video = concatenate_videoclips(clips, method="compose", padding=padding)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Rendering CPU video to %s", output_path)
            video.write_videofile(
                str(output_path),
                fps=settings.video_fps,
                codec="libx264",
                audio=False,
                preset="medium",
                threads=os.cpu_count() or 4,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                logger=None,
            )
            return output_path
        except Exception as exc:
            raise VideoServiceError(f"MoviePy failed to render the reel: {exc}") from exc
        finally:
            for clip in clips:
                try:
                    clip.close()
                except Exception:
                    pass
