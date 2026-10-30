from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx
import websockets

from config import settings

logger = logging.getLogger(__name__)

POSITIVE_TITLE_HINTS = (
    "positive",
    "prompt",
    "clip text encode (prompt)",
)


class ComfyServiceError(RuntimeError):
    pass


class ComfyService:
    def __init__(self) -> None:
        self.http_url = settings.comfy_http_url.rstrip("/")
        self.ws_url = settings.comfy_ws_url
        self.workflow_path = settings.workflow_path

    async def generate_frames(self, prompts: list[str], job_dir: Path) -> list[Path]:
        workflow_template = self._load_workflow()
        job_dir.mkdir(parents=True, exist_ok=True)
        saved_frames: list[Path] = []

        for index, prompt in enumerate(prompts, start=1):
            logger.info("ComfyUI frame %s/%s", index, len(prompts))
            image_bytes, filename = await self._generate_one(workflow_template, prompt, index)
            frame_path = job_dir / f"frame_{index:02d}{Path(filename).suffix or '.png'}"
            frame_path.write_bytes(image_bytes)
            saved_frames.append(frame_path)

        return saved_frames

    def _load_workflow(self) -> dict[str, Any]:
        if not self.workflow_path.exists():
            raise ComfyServiceError(f"Missing workflow file: {self.workflow_path}")
        try:
            workflow = json.loads(self.workflow_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ComfyServiceError("workflow_api.json is not valid JSON.") from exc

        if not isinstance(workflow, dict) or not workflow:
            raise ComfyServiceError("workflow_api.json must be a non-empty object.")
        return workflow

    async def _generate_one(
        self,
        workflow_template: dict[str, Any],
        english_prompt: str,
        frame_index: int,
    ) -> tuple[bytes, str]:
        client_id = str(uuid.uuid4())
        workflow = copy.deepcopy(workflow_template)
        self._inject_positive_prompt(workflow, english_prompt)
        self._randomize_seed(workflow, frame_index)

        prompt_id = await self._queue_prompt(workflow, client_id)
        await self._wait_until_complete(client_id, prompt_id)
        return await self._download_output_image(prompt_id)

    def _inject_positive_prompt(self, workflow: dict[str, Any], english_prompt: str) -> None:
        target_id = self._find_positive_prompt_node(workflow)
        node = workflow[target_id]
        inputs = node.setdefault("inputs", {})
        if "text" not in inputs:
            raise ComfyServiceError(
                f"Positive prompt node {target_id} has no 'text' input. "
                "Export workflow_api.json from ComfyUI again."
            )
        inputs["text"] = english_prompt
        logger.debug("Injected prompt into node %s", target_id)

    def _find_positive_prompt_node(self, workflow: dict[str, Any]) -> str:
        clip_nodes: list[tuple[str, dict[str, Any]]] = []
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") == "CLIPTextEncode":
                clip_nodes.append((str(node_id), node))

        for node_id, node in clip_nodes:
            title = str(node.get("_meta", {}).get("title", "")).lower()
            if any(hint in title for hint in POSITIVE_TITLE_HINTS) and "negative" not in title:
                return node_id

        for node_id, node in clip_nodes:
            title = str(node.get("_meta", {}).get("title", "")).lower()
            if "negative" not in title:
                return node_id

        if clip_nodes:
            return clip_nodes[0][0]

        raise ComfyServiceError(
            "No CLIPTextEncode node found in workflow_api.json. "
            "Export the API-format workflow from ComfyUI."
        )

    def _randomize_seed(self, workflow: dict[str, Any], frame_index: int) -> None:
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or "seed" not in inputs:
                continue
            try:
                base_seed = int(inputs["seed"])
            except (TypeError, ValueError):
                base_seed = 42
            inputs["seed"] = base_seed + frame_index * 97

    async def _queue_prompt(self, workflow: dict[str, Any], client_id: str) -> str:
        payload = {"prompt": workflow, "client_id": client_id}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.http_url}/prompt", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise ComfyServiceError(
                "Cannot connect to ComfyUI at 127.0.0.1:8188. Start ComfyUI first."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise ComfyServiceError(f"ComfyUI rejected the workflow: {detail}") from exc

        prompt_id = data.get("prompt_id")
        node_errors = data.get("node_errors") or {}
        if node_errors:
            raise ComfyServiceError(f"ComfyUI node errors: {node_errors}")
        if not prompt_id:
            raise ComfyServiceError("ComfyUI did not return a prompt_id.")
        return str(prompt_id)

    async def _wait_until_complete(self, client_id: str, prompt_id: str) -> None:
        uri = f"{self.ws_url}?clientId={client_id}"
        try:
            async with websockets.connect(uri, max_size=None, open_timeout=15) as websocket:
                deadline = asyncio.get_event_loop().time() + settings.comfy_timeout_seconds
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        raise ComfyServiceError("Timed out waiting for ComfyUI to finish.")
                    raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                    if isinstance(raw, bytes):
                        continue
                    message = json.loads(raw)
                    message_type = message.get("type")
                    data = message.get("data") or {}

                    if message_type == "execution_error" and data.get("prompt_id") == prompt_id:
                        raise ComfyServiceError(
                            f"ComfyUI execution error: {data.get('exception_message') or data}"
                        )

                    if (
                        message_type == "executing"
                        and data.get("prompt_id") == prompt_id
                        and data.get("node") is None
                    ):
                        return
        except ComfyServiceError:
            raise
        except (OSError, websockets.WebSocketException) as exc:
            raise ComfyServiceError(
                "ComfyUI WebSocket connection failed. Is ComfyUI running on port 8188?"
            ) from exc
        except asyncio.TimeoutError as exc:
            raise ComfyServiceError("Timed out waiting for ComfyUI to finish.") from exc

    async def _download_output_image(self, prompt_id: str) -> tuple[bytes, str]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                history_response = await client.get(f"{self.http_url}/history/{prompt_id}")
                history_response.raise_for_status()
                history = history_response.json()
        except httpx.HTTPError as exc:
            raise ComfyServiceError("Could not read ComfyUI history after generation.") from exc

        job = history.get(prompt_id) or {}
        outputs = job.get("outputs") or {}
        image_meta = self._first_image_meta(outputs)
        if image_meta is None:
            raise ComfyServiceError("ComfyUI finished but no image was found in the output nodes.")

        params = {
            "filename": image_meta["filename"],
            "subfolder": image_meta.get("subfolder", ""),
            "type": image_meta.get("type", "output"),
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                view_response = await client.get(f"{self.http_url}/view", params=params)
                view_response.raise_for_status()
                return view_response.content, str(image_meta["filename"])
        except httpx.HTTPError as exc:
            raise ComfyServiceError("ComfyUI saved a frame but downloading it failed.") from exc

    def _first_image_meta(self, outputs: dict[str, Any]) -> dict[str, Any] | None:
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            images = node_output.get("images") or []
            for image in images:
                if isinstance(image, dict) and image.get("filename"):
                    folder_type = str(image.get("type") or "output")
                    if folder_type == "output":
                        return image
            if images and isinstance(images[0], dict) and images[0].get("filename"):
                return images[0]
        return None
