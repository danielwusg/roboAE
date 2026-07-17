from __future__ import annotations

import base64
import copy
import io
import json
import os
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import requests
from PIL import Image

from robot_auto_evolve.agent import (
    Detection,
    DetectionRequest,
    DetectionResult,
    LanguageRequest,
    PointingRequest,
    PointingResult,
    SegmentationRequest,
    SegmentationResult,
    TextResult,
    VisionRequest,
)
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.services import ServiceIdentity


class ToolBackend(ABC):
    def __init__(self, identity: ServiceIdentity) -> None:
        self.identity = identity

    def load(self) -> None:
        return

    def smoke(self) -> None:
        return

    @abstractmethod
    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError


def _pil(value: Any, path: str = "image") -> Image.Image:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise StrictSchemaError(f"{path}: expected uint8 RGB array")
    return Image.fromarray(value, mode="RGB")


def _data_url(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    _pil(image).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _generation_options(model: Any, max_new_tokens: int, temperature: float = 0.0) -> dict[str, Any]:
    config = copy.deepcopy(model.generation_config)
    if hasattr(config, "_from_model_config"):
        config._from_model_config = False
    config.do_sample = temperature > 0
    if temperature > 0:
        config.temperature = temperature
    else:
        for name in ("temperature", "top_p", "top_k", "min_p", "typical_p"):
            if hasattr(config, name):
                setattr(config, name, None)
    return {"max_new_tokens": max_new_tokens, "generation_config": config, "use_model_defaults": False}


class FixtureBackend(ToolBackend):
    def __init__(self, identity: ServiceIdentity, fixture_path: Path) -> None:
        super().__init__(identity)
        self.responses = json.loads(fixture_path.read_text())

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        del payload
        result = self.responses.get(operation)
        if not isinstance(result, Mapping):
            raise RuntimeError(f"fixture has no mapping response for {operation}")
        return dict(result)


class OpenAICompatibleBackend(ToolBackend):
    def __init__(
        self,
        identity: ServiceIdentity,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_s: float,
    ) -> None:
        super().__init__(identity)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _request(self, messages: list[dict[str, Any]], max_tokens: int, temperature: float) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        obj = response.json()
        try:
            content = obj["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenAI-compatible server returned an invalid response") from exc
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, Mapping))
        if type(content) is not str or not content:
            raise RuntimeError("OpenAI-compatible server returned empty text")
        return content

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation == "generate" and self.identity.service_kind == "language":
            request = LanguageRequest.from_mapping(payload)
            prompt = "\n".join([*request.context, request.instruction])
            text = self._request(
                [{"role": "user", "content": prompt}],
                request.max_tokens,
                request.temperature,
            )
            return TextResult(text).to_mapping()
        if operation == "describe" and self.identity.service_kind == "vision":
            request = VisionRequest.from_mapping(payload)
            content: list[dict[str, Any]] = [
                {"type": "text", "text": "\n".join([*request.context, request.instruction])}
            ]
            for _, image in sorted(request.images.items()):
                content.append({"type": "image_url", "image_url": {"url": _data_url(image)}})
            text = self._request(
                [{"role": "user", "content": content}],
                request.max_tokens,
                0.0,
            )
            return TextResult(text).to_mapping()
        raise StrictSchemaError(f"{self.identity.service_name}: unsupported operation {operation}")

    def smoke(self) -> None:
        if self.identity.service_kind == "language":
            self.invoke("generate", LanguageRequest("Reply OK.", max_tokens=2).to_mapping())
        else:
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            self.invoke("describe", VisionRequest("Reply OK.", {"smoke": image}, max_tokens=2).to_mapping())


class TransformersLanguageBackend(ToolBackend):
    def __init__(self, identity: ServiceIdentity, device: str) -> None:
        super().__init__(identity)
        self.device = device
        self._processor: Any = None
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._processor = AutoTokenizer.from_pretrained(
            self.identity.model_id,
            revision=self.identity.checkpoint_revision,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.identity.model_id,
            revision=self.identity.checkpoint_revision,
            dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        ).eval()

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation != "generate":
            raise StrictSchemaError(f"{self.identity.service_name}: unsupported operation {operation}")
        self.load()
        request = LanguageRequest.from_mapping(payload)
        prompt = "\n".join([*request.context, request.instruction])
        messages = [{"role": "user", "content": prompt}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text, return_tensors="pt").to(self._model.device)
        output = self._model.generate(
            **inputs,
            **_generation_options(self._model, request.max_tokens, request.temperature),
        )
        generated = output[:, inputs.input_ids.shape[1] :]
        return TextResult(self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()).to_mapping()

    def smoke(self) -> None:
        self.invoke("generate", LanguageRequest("Reply OK.", max_tokens=2).to_mapping())


class TransformersVisionBackend(ToolBackend):
    def __init__(self, identity: ServiceIdentity, device: str) -> None:
        super().__init__(identity)
        self.device = device
        self._processor: Any = None
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.identity.model_id,
            revision=self.identity.checkpoint_revision,
            trust_remote_code=True,
        )
        kwargs = {
            "revision": self.identity.checkpoint_revision,
            "dtype": torch.bfloat16,
            "device_map": self.device,
            "trust_remote_code": True,
        }
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(self.identity.model_id, **kwargs).eval()
        except (ValueError, ImportError):
            self._model = AutoModelForCausalLM.from_pretrained(self.identity.model_id, **kwargs).eval()

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation != "describe":
            raise StrictSchemaError(f"{self.identity.service_name}: unsupported operation {operation}")
        self.load()
        request = VisionRequest.from_mapping(payload)
        images = [_pil(image) for _, image in sorted(request.images.items())]
        text = "\n".join([*request.context, request.instruction])
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=images, return_tensors="pt").to(self._model.device)
        output = self._model.generate(**inputs, **_generation_options(self._model, request.max_tokens))
        input_length = inputs["input_ids"].shape[1]
        generated = output[:, input_length:]
        result = self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return TextResult(result).to_mapping()

    def smoke(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.invoke("describe", VisionRequest("Reply OK.", {"smoke": image}, max_tokens=2).to_mapping())


class GroundingDinoBackend(ToolBackend):
    def __init__(self, identity: ServiceIdentity, device: str) -> None:
        super().__init__(identity)
        self.device = device
        self._processor: Any = None
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.identity.model_id,
            revision=self.identity.checkpoint_revision,
        )
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.identity.model_id,
            revision=self.identity.checkpoint_revision,
        ).to(self.device).eval()

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation != "detect":
            raise StrictSchemaError(f"{self.identity.service_name}: unsupported operation {operation}")
        self.load()
        import torch

        request = DetectionRequest.from_mapping(payload)
        image = _pil(request.image)
        query = request.query.lower().strip().rstrip(".") + "."
        inputs = self._processor(images=image, text=query, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=request.threshold,
            text_threshold=request.threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = results.get("text_labels")
        if labels is None:
            labels = results.get("labels", [])
        detections = tuple(
            Detection(str(label), float(score), tuple(float(x) for x in box.tolist()))
            for box, score, label in zip(results["boxes"], results["scores"], labels, strict=True)
        )
        return DetectionResult(detections).to_mapping()

    def smoke(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.invoke("detect", DetectionRequest(image, "object").to_mapping())


class Sam3Backend(ToolBackend):
    MODEL_RESOLUTION = 1008
    PROMPT_ROUNDOFF_TOLERANCE_PX = 0.01
    SAFETENSORS_VERSION = "0.8.0"

    def __init__(self, identity: ServiceIdentity, device: str, checkpoint_sha256: str) -> None:
        super().__init__(identity)
        self.device = device
        self.checkpoint_sha256 = checkpoint_sha256
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        import hashlib

        import safetensors
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if safetensors.__version__ != self.SAFETENSORS_VERSION:
            raise RuntimeError(f"SAM3 requires safetensors {self.SAFETENSORS_VERSION}")

        checkpoint = Path(
            hf_hub_download(
                repo_id=self.identity.model_id,
                filename="sam3.safetensors",
                revision=self.identity.checkpoint_revision,
                local_files_only=True,
            )
        )
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != self.checkpoint_sha256:
            raise RuntimeError("SAM3 checkpoint SHA-256 mismatch")
        source_state = load_file(checkpoint, device="cpu")
        prefixes = {key.split(".", 1)[0] for key in source_state}
        if prefixes != {"detector", "tracker"}:
            raise RuntimeError(f"SAM3 checkpoint has unexpected key prefixes: {sorted(prefixes)}")
        model = build_sam3_image_model(
            device="cpu",
            checkpoint_path=None,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,
            compile=False,
        )
        mapped_state = {
            (key.removeprefix("detector.") if key.startswith("detector.") else "inst_interactive_predictor.model." + key.removeprefix("tracker.")): value
            for key, value in source_state.items()
        }
        missing, unexpected = model.load_state_dict(mapped_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"SAM3 state mismatch: missing={len(missing)} {sorted(missing)[:20]}, "
                f"unexpected={len(unexpected)} {sorted(unexpected)[:20]}"
            )
        del source_state, mapped_state
        self._model = model.to(self.device).eval()
        if self._model.inst_interactive_predictor is None:
            raise RuntimeError("SAM3 interactive predictor is unavailable")
        self._processor = Sam3Processor(
            self._model,
            resolution=self.MODEL_RESOLUTION,
            device=self.device,
        )

    @staticmethod
    def _prepare_prompts(
        request: SegmentationRequest,
    ) -> tuple[tuple[tuple[float, float, float, float], ...], np.ndarray | None, np.ndarray | None]:
        height, width = request.image.shape[:2]
        boxes = []
        for x0, y0, x1, y1 in request.boxes_xyxy:
            tolerance = Sam3Backend.PROMPT_ROUNDOFF_TOLERANCE_PX
            if not (-tolerance <= x0 < x1 <= width + tolerance and -tolerance <= y0 < y1 <= height + tolerance):
                raise StrictSchemaError("segmentation_request.box: expected in-bounds xyxy")
            clipped = (
                min(float(width), max(0.0, x0)),
                min(float(height), max(0.0, y0)),
                min(float(width), max(0.0, x1)),
                min(float(height), max(0.0, y1)),
            )
            if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                raise StrictSchemaError("segmentation_request.box: expected nondegenerate xyxy")
            boxes.append(clipped)
        for x, y in request.points_xy:
            if not (0.0 <= x < width and 0.0 <= y < height):
                raise StrictSchemaError("segmentation_request.point: expected in-bounds xy")
        points = np.asarray(request.points_xy, dtype=np.float32) if request.points_xy else None
        labels = np.asarray(request.labels, dtype=np.int32) if request.points_xy else None
        return tuple(boxes), points, labels

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation != "segment":
            raise StrictSchemaError(f"{self.identity.service_name}: unsupported operation {operation}")
        self.load()
        import torch

        request = SegmentationRequest.from_mapping(payload)
        prepared_boxes, points, labels = self._prepare_prompts(request)
        boxes = prepared_boxes or (None,)
        masks_out: list[np.ndarray] = []
        scores_out: list[float] = []
        with self._lock, torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda"),
        ):
            state = self._processor.set_image(_pil(request.image))
            for box in boxes:
                masks, scores, _ = self._model.predict_inst(
                    state,
                    point_coords=points,
                    point_labels=labels,
                    box=None if box is None else np.asarray(box, dtype=np.float32),
                    multimask_output=True,
                    return_logits=False,
                    normalize_coords=True,
                )
                scores = np.asarray(scores, dtype=np.float32)
                if masks.ndim != 3 or scores.shape != (masks.shape[0],) or not np.all(np.isfinite(scores)):
                    raise RuntimeError("SAM3 returned invalid mask scores")
                index = int(scores.argmax())
                score = float(scores[index])
                if not -1e-6 <= score <= 1.0 + 1e-6:
                    raise RuntimeError("SAM3 returned an out-of-range mask score")
                masks_out.append(np.asarray(masks[index], dtype=np.bool_))
                scores_out.append(min(1.0, max(0.0, score)))
        result = SegmentationResult(
            np.stack(masks_out, axis=0),
            np.asarray(scores_out, dtype=np.float32),
        )
        return result.to_mapping()

    def smoke(self) -> None:
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[32:96, 32:96] = 255
        result = SegmentationResult.from_mapping(
            self.invoke(
                "segment",
                SegmentationRequest(image, boxes_xyxy=((24.0, 24.0, 104.0, 104.0),)).to_mapping(),
            )
        )
        if result.masks.shape != (1, 128, 128):
            raise RuntimeError("SAM3 smoke returned an invalid mask shape")


class RoboPointBackend(ToolBackend):
    _POINT = re.compile(r"(?:x|x\d+)\s*=\s*[\"']?([0-9.]+).*?(?:y|y\d+)\s*=\s*[\"']?([0-9.]+)", re.I)

    def __init__(self, identity: ServiceIdentity, device: str) -> None:
        super().__init__(identity)
        self.device = device
        self._processor: Any = None
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.identity.model_id,
            revision=self.identity.checkpoint_revision,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.identity.model_id,
            revision=self.identity.checkpoint_revision,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation != "point":
            raise StrictSchemaError(f"{self.identity.service_name}: unsupported operation {operation}")
        self.load()
        request = PointingRequest.from_mapping(payload)
        image = _pil(request.image)
        prompt = f"Point to the image location needed to: {request.instruction}"
        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(self._model.device)
        output = self._model.generate(**inputs, **_generation_options(self._model, 128))
        generated = output[:, inputs["input_ids"].shape[1] :]
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        width, height = image.size
        points = []
        for raw_x, raw_y in self._POINT.findall(text):
            x, y = float(raw_x), float(raw_y)
            if x <= 1.0 and y <= 1.0:
                x, y = x * width, y * height
            elif x <= 100.0 and y <= 100.0:
                x, y = x * width / 100.0, y * height / 100.0
            points.append((x, y))
        if not points:
            raise RuntimeError(f"RoboPoint returned no parseable points: {text[:300]}")
        return PointingResult(tuple(points), tuple(1.0 for _ in points)).to_mapping()

    def smoke(self) -> None:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        self.invoke("point", PointingRequest(image, "point to the center").to_mapping())


def api_key_from_environment(variable: str | None) -> str | None:
    if variable is None:
        return None
    return os.environ.get(variable)
