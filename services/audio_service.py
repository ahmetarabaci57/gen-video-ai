import os
import logging
from pathlib import Path
from gtts import gTTS

logger = logging.getLogger(__name__)

class AudioService:
    def generate_turkish_audio(self, text: str, output_path: Path) -> Path:
        """CPU üzerinde tamamen ücretsiz Türkçe ses üretir ve kaydeder."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Generating local gTTS audio for text: %s...", text[:30])
            
            tts = gTTS(text=text, lang='tr', slow=False)
            tts.save(str(output_path))
            
            logger.info("Audio successfully saved to %s", output_path)
            return output_path
        except Exception as e:
            logger.error("Audio generation failed: %s", e)
            raise RuntimeError(f"Ses üretim hatası: {e}")
