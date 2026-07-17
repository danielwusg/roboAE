from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_auto_evolve.agent import PointingRequest, PointingResult, TextResult, VisionRequest
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.services import ServiceIdentity

from .backends import ToolBackend, _pil


_COORDS = re.compile(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"/?>')
_FRAME = re.compile(r"(?:^|\t|:|,|;)([0-9.]+) ([0-9. ]+)")
_POINT = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
_NO_POINTS = frozenset({"none", "there are none", "there is none"})


def parse_image_points(text: str, width: int, height: int) -> tuple[tuple[float, float], ...]:
    if type(text) is not str or type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("invalid Molmo2 point parser input")
    result: list[tuple[float, float]] = []
    for coordinates in _COORDS.finditer(text):
        for frame in _FRAME.finditer(coordinates.group(1)):
            if float(frame.group(1)) != 1.0:
                continue
            for point in _POINT.finditer(frame.group(2)):
                x = float(point.group(2)) * width / 1000.0
                y = float(point.group(3)) * height / 1000.0
                if 0.0 <= x <= width and 0.0 <= y <= height:
                    result.append((x, y))
    return tuple(result)


class Molmo2Backend(ToolBackend):
    TRANSFORMERS_VERSION = "4.57.1"
    TORCH_DTYPE = "bfloat16"
    POINT_MAX_TOKENS = 256
    POINT_CONFIDENCE = 1.0

    def __init__(self, identity: ServiceIdentity, device: str) -> None:
        super().__init__(identity)
        if identity.service_kind not in {"vision", "pointing"}:
            raise ValueError("Molmo2 requires a vision or pointing identity")
        self.device = device
        self._processor: Any = None
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if transformers.__version__ != self.TRANSFORMERS_VERSION:
            raise RuntimeError(f"Molmo2 requires transformers {self.TRANSFORMERS_VERSION}")
        snapshot = Path(
            snapshot_download(
                repo_id=self.identity.model_id,
                revision=self.identity.checkpoint_revision,
                local_files_only=True,
            )
        )
        if snapshot.name != self.identity.checkpoint_revision:
            raise RuntimeError("Molmo2 resolved snapshot revision mismatch")
        common = {"trust_remote_code": True, "local_files_only": True}
        self._processor = AutoProcessor.from_pretrained(
            snapshot,
            padding_side="left",
            **common,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            snapshot,
            dtype=torch.bfloat16,
            device_map=self.device,
            **common,
        ).eval()

    def _generate(self, instruction: str, images: list[Any], max_tokens: int) -> str:
        self.load()
        import torch

        content = [{"type": "text", "text": instruction}]
        content.extend({"type": "image", "image": image} for image in images)
        inputs = self._processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )
        inputs = {name: value.to(self._model.device) for name, value in inputs.items()}
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda"),
        ):
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_model_defaults=False,
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        text = self._processor.post_process_image_text_to_text(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if not text:
            raise RuntimeError("Molmo2 returned empty text")
        return text

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation == "describe" and self.identity.service_kind == "vision":
            request = VisionRequest.from_mapping(payload)
            instruction = "\n".join([*request.context, request.instruction])
            images = [_pil(image) for _, image in sorted(request.images.items())]
            return TextResult(self._generate(instruction, images, request.max_tokens)).to_mapping()
        if operation == "point" and self.identity.service_kind == "pointing":
            request = PointingRequest.from_mapping(payload)
            image = _pil(request.image)
            text = self._generate(
                f"Point to the image location needed to: {request.instruction}",
                [image],
                self.POINT_MAX_TOKENS,
            )
            points = parse_image_points(text, image.width, image.height)
            if not points:
                normalized = text.strip().rstrip(".!?").strip().casefold()
                if normalized in _NO_POINTS:
                    return PointingResult((), ()).to_mapping()
                raise RuntimeError(f"Molmo2 returned no canonical image points: {text[:300]}")
            return PointingResult(points, tuple(self.POINT_CONFIDENCE for _ in points)).to_mapping()
        raise StrictSchemaError(f"{self.identity.service_name}: unsupported operation {operation}")

    def smoke(self) -> None:
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[80:176, 80:176, 0] = 255
        if self.identity.service_kind == "vision":
            self.invoke(
                "describe",
                VisionRequest("Describe the central red shape briefly.", {"main": image}, max_tokens=24).to_mapping(),
            )
        else:
            self.invoke("point", PointingRequest(image, "locate the central red square").to_mapping())
