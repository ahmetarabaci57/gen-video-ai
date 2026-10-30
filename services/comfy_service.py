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
        """
        Gelen prompt listesindeki ilk elemanı alır ve Windows dosya sistemiyle 
        çakışmayacak şekilde temiz ve güvenli bir isimle yerel diske kaydeder.
        """
        workflow_template = self._load_workflow()
        job_dir.mkdir(parents=True, exist_ok=True)
        saved_frames: list[Path] = []

        if not prompts:
            raise ComfyServiceError("ComfyUI'a gönderilecek herhangi bir prompt bulunamadı.")

        # Listenin içindeki ilk promptu güvenle ayıklıyoruz
        target_prompt = prompts[0] if isinstance(prompts, list) else str(prompts)
        
        logger.info("ComfyUI tek kare çizimi başlatılıyor...")
        logger.info(target_prompt)
        
        # ComfyUI'dan resmi ürettiriyoruz
        image_bytes, _ = await self._generate_one(workflow_template, target_prompt, 1)
        logger.info("image generated")
        # 🚨 [Errno 22] ÇÖZÜMÜ: Gelen 'filename' değişkenine asla güvenmiyoruz!
        # Doğrudan kendimiz güvenli ve sabit bir dosya adı (.png) veriyoruz.
        frame_path = job_dir / "frame_01.png"
        
        # Dosyayı diske yazıyoruz
        frame_path.write_bytes(image_bytes)
        saved_frames.append(frame_path)
        
        logger.info("Görsel Windows dosya sistemine güvenle kaydedildi: %s", frame_path)
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
        
        # 🚨 [Errno 22] ENGELLEYİCİ: prompt metnini saf string'e zorluyoruz
        # İçindeki olası gizli dizi/liste karakterlerini tamamen temizliyoruz
        clean_prompt = str(english_prompt).replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
        
        self._inject_positive_prompt(workflow, clean_prompt)
        self._randomize_seed(workflow, frame_index)
        logger.info("random seed injected")
        prompt_id = await self._queue_prompt(workflow, client_id)
        logger.info("prompt queued")
        await self._wait_until_complete(client_id, prompt_id)
        logger.info("prompt completed")
        return await self._download_output_image(prompt_id)


    def _inject_positive_prompt(self, workflow: dict[str, Any], english_prompt: str) -> None:
        target_id = self._find_positive_prompt_node(workflow)
        logger.info("positive prompt node : " + target_id )
        node = workflow[target_id]
        inputs = node.setdefault("inputs", {})
        if "text" not in inputs:
            raise ComfyServiceError(
                f"Positive prompt node {target_id} has no 'text' input. "
                "Export workflow_api.json from ComfyUI again."
            )
        inputs["text"] = str(english_prompt)
        logger.info("ComfyUI Node ID [%s] alanına prompt enjekte edildi.", target_id)

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

        raise ComfyServiceError("workflow_api.json içinde CLIPTextEncode düğümü bulunamadı.")

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
        """
        [Errno 22] tqdm/flush hatasını engellemek için ComfyUI KSampler 
        düğümlerindeki ilerleme durum çubuğu çizimini kodla kapatıyoruz.
        """
        # Workflow içindeki tüm KSampler düğümlerini tarayıp terminal pbar'ını susturuyoruz
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict) and node_data.get("class_type") == "KSampler":
                inputs = node_data.setdefault("inputs", {})
                # tqdm çakışmasını engellemek için gizli parametreyi True yapıyoruz
                inputs["disable_pbar"] = True 

        payload = {"prompt": workflow, "client_id": client_id}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.http_url}/prompt", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise ComfyServiceError("ComfyUI bağlantısı kurulamadı. Port: 8188 açık mı?") from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise ComfyServiceError(f"ComfyUI akışı reddetti: {detail}") from exc

        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyServiceError("ComfyUI bir prompt_id döndüremedi.")
        return str(prompt_id)


    async def _wait_until_complete(self, client_id: str, prompt_id: str) -> None:
        uri = f"{self.ws_url}?clientId={client_id}"
        try:
            async with websockets.connect(uri, max_size=None, open_timeout=15) as websocket:
                deadline = asyncio.get_event_loop().time() + settings.comfy_timeout_seconds
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        raise ComfyServiceError("ComfyUI zaman aşımına uğradı.")
                    raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                    if isinstance(raw, bytes):
                        continue
                    message = json.loads(raw)
                    message_type = message.get("type")
                    data = message.get("data") or {}

                    if message_type == "execution_error" and data.get("prompt_id") == prompt_id:
                        # 🚨 [Errno 22] ÇÖZÜMÜ: Ham nesneyi veya kaçış karakteri barındıran stringleri fırlatmak yerine
                        # sadece en yalın, temiz metni ayıklayıp hata olarak fırlatıyoruz.
                        raw_error = str(data.get('exception_message') or "Unknown ComfyUI Node Error")
                        clean_error_msg = raw_error.replace("\\", "/").replace("'", "").replace('"', "")
                        
                        logger.error("!!! COMFYUI GERÇEK İÇ HATASI !!!: %s", clean_error_msg)
                        raise ComfyServiceError(f"ComfyUI Dahili Hatasi: {clean_error_msg}")


                    if (
                        message_type == "executing"
                        and data.get("prompt_id") == prompt_id
                        and data.get("node") is None
                    ):
                        return
        except ComfyServiceError:
            raise
        except Exception as exc:
            raise ComfyServiceError(f"WebSocket dinleme aşaması başarısız: {exc}") from exc

    async def _download_output_image(self, prompt_id: str) -> tuple[bytes, str]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            history_response = await client.get(f"{self.http_url}/history/{prompt_id}")
            history = history_response.json()

        job = history.get(prompt_id) or {}
        outputs = job.get("outputs") or {}
        image_meta = self._first_image_meta(outputs)
        if image_meta is None:
            raise ComfyServiceError("İşlem bitti ama çıktı düğümlerinde görsel bulunamadı.")

        params = {
            "filename": image_meta["filename"],
            "subfolder": image_meta.get("subfolder", ""),
            "type": image_meta.get("type", "output"),
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            view_response = await client.get(f"{self.http_url}/view", params=params)
            return view_response.content, str(image_meta["filename"])

    def _first_image_meta(self, outputs: dict[str, Any]) -> dict[str, Any] | None:
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            images = node_output.get("images") or []
            for image in images:
                if isinstance(image, dict) and image.get("filename"):
                    return image
        return None
