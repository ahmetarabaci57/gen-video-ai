from __future__ import annotations

import json
import logging
import re

import httpx

from config import settings

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """You are a prompt engineer for Stable Diffusion 1.5.
The user will give you an Instagram Reels idea in Turkish.
Translate the idea into English, then expand it into detailed visual scene prompts.

Rules:
- Output ONLY a JSON array of strings. No markdown, no commentary.
- Array length MUST equal the requested frame count.
- Each string is one English image prompt for a vertical 9:16 Instagram Reel frame (1080x1920).
- Keep character, location, lighting, and style consistent across frames so they feel like one reel.
- Include: subject, action, camera angle, lighting, mood, clothing, environment, photorealistic details.
- Always append: "vertical composition, 9:16 aspect ratio, cinematic still, highly detailed, sharp focus, instagram reel frame"
- Do NOT mention text, watermarks, logos, or UI.
- Do NOT write Turkish in the prompts.
"""


class LLMServiceError(RuntimeError):
    pass


PREFERRED_MODELS = (
    "llama3.2:3b",
    "llama3.2",
    "llama3.1:latest",
    "llama3.1",
    "llama3",
)


class LLMService:
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    async def idea_to_image_prompts(self, turkish_idea: str, num_frames: int) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
                model = await self._resolve_model(client)
                payload = {
                    "model": model,
                    "prompt": (
                        f"{SYSTEM_INSTRUCTIONS}\n\n"
                        f"Frame count: {num_frames}\n"
                        f"Turkish idea: {turkish_idea.strip()}\n"
                    ),
                    "stream": False,
                    "keep_alive": 0,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 900,
                    },
                }
                logger.info("Using Ollama model %s", model)
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                if response.status_code == 404:
                    raise LLMServiceError(self._ollama_http_message(response))
                response.raise_for_status()
                data = response.json()
        except LLMServiceError:
            raise
        except httpx.ConnectError as exc:
            raise LLMServiceError(
                "Cannot connect to Ollama at 127.0.0.1:11434. "
                "Start Ollama and pull a text model first."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMServiceError("Ollama timed out while translating the idea.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMServiceError(self._ollama_http_message(exc.response)) from exc
        except json.JSONDecodeError as exc:
            raise LLMServiceError("Ollama returned a non-JSON response.") from exc

        raw_text = (data.get("response") or "").strip()
        if not raw_text:
            raise LLMServiceError("Ollama returned an empty prompt.")

        prompts = self._parse_prompts(raw_text, num_frames)
        logger.info("Ollama produced %s frame prompts", len(prompts))
        return prompts

    async def _resolve_model(self, client: httpx.AsyncClient) -> str:
        names = await self._list_models(client)
        if not names:
            raise LLMServiceError(
                "Ollama has no models installed. Run: ollama pull llama3.2:3b"
            )

        configured = self.model.strip()
        if configured in names:
            return configured

        for candidate in PREFERRED_MODELS:
            if candidate in names:
                logger.warning(
                    "Configured model %s is not installed. Falling back to %s. Installed: %s",
                    configured,
                    candidate,
                    ", ".join(names),
                )
                return candidate

        fallback = names[0]
        logger.warning(
            "Configured model %s is not installed. Falling back to %s. Installed: %s",
            configured,
            fallback,
            ", ".join(names),
        )
        return fallback

    async def _list_models(self, client: httpx.AsyncClient) -> list[str]:
        try:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise LLMServiceError("Could not list Ollama models from /api/tags.") from exc

        models = payload.get("models") or []
        names: list[str] = []
        for item in models:
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                names.append(name)
        return names

    def _ollama_http_message(self, response: httpx.Response) -> str:
        detail = response.text.strip()
        try:
            payload = response.json()
            detail = str(payload.get("error") or payload)
        except json.JSONDecodeError:
            pass
        return f"Ollama HTTP {response.status_code}: {detail}"

    def _parse_prompts(self, raw_text: str, num_frames: int) -> list[str]:
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        prompts: list[str] = []
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                prompts = [str(item).strip() for item in parsed if str(item).strip()]
            elif isinstance(parsed, dict):
                for key in ("prompts", "frames", "scenes"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        prompts = [str(item).strip() for item in value if str(item).strip()]
                        break
        except json.JSONDecodeError:
            prompts = []

        if not prompts:
            numbered = re.findall(r"(?:^|\n)\s*(?:\d+[\).:-]|-)\s*(.+)", cleaned)
            prompts = [item.strip(" \"'") for item in numbered if item.strip()]

        if not prompts:
            fallback = " ".join(cleaned.split())
            prompts = [fallback] if fallback else []

        if not prompts:
            raise LLMServiceError("Could not parse English image prompts from Ollama.")

        suffix = (
            "vertical composition, 9:16 aspect ratio, cinematic still, "
            "highly detailed, sharp focus, instagram reel frame"
        )
        normalized: list[str] = []
        for prompt in prompts:
            text = prompt.strip().strip('"')
            if "9:16" not in text:
                text = f"{text}, {suffix}"
            normalized.append(text)

        while len(normalized) < num_frames:
            normalized.append(normalized[-1])
        return normalized[:num_frames]
