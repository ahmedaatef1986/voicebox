"""Native VoiceTut-TTS backend for Egyptian Arabic preset voices."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base import combine_voice_prompts as _combine_voice_prompts
from .base import get_torch_device, is_model_cached, model_load_progress

logger = logging.getLogger(__name__)

VOICETUT_HF_REPO = "mohammedaly22/VoiceTut-TTS"
VOICETUT_DEFAULT_SPEAKER = "Mohamed"
VOICETUT_SAMPLE_RATE = 24000
VOICETUT_MODEL_PATTERNS = [
    "config.json",
    "chat_template.jinja",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "reference_speakers/*",
]

# (speaker_id, display_name, gender, native_language_code, description)
VOICETUT_VOICES = [
    ("Abdelrahman", "Abdelrahman", "male", "ar", "Professional, energetic Egyptian voice"),
    ("Abdullah", "Abdullah", "male", "ar", "Formal, professional Egyptian voice"),
    ("Kamal", "Kamal", "male", "ar", "Serious, clear Egyptian voice"),
    ("Hossam", "Hossam", "male", "ar", "Warm, friendly Egyptian voice"),
    ("Mohamed", "Mohamed", "male", "ar", "Calm, deep Egyptian voice"),
    ("Omar", "Omar", "male", "ar", "Storytelling, measured Egyptian voice"),
    ("Sayed", "Sayed", "male", "ar", "Youthful, expressive Egyptian voice"),
    ("Zaki", "Zaki", "male", "ar", "Composed, trustworthy Egyptian voice"),
    ("Aly", "Aly", "male", "ar", "Calm, proud Egyptian voice"),
    ("Essam", "Essam", "male", "ar", "Serious, clear Egyptian voice"),
    ("Ahmed", "Ahmed", "male", "ar", "Youthful, lively Egyptian voice"),
    ("Asmaa", "Asmaa", "female", "ar", "Soft, friendly Egyptian voice"),
    ("Esraa", "Esraa", "female", "ar", "Cheerful, lively Egyptian voice"),
    ("Hanan", "Hanan", "female", "ar", "Warm, calm Egyptian voice"),
    ("Sarah", "Sarah", "female", "ar", "Elegant, maternal Egyptian voice"),
    ("Yasmin", "Yasmin", "female", "ar", "Youthful, bright Egyptian voice"),
    ("Omnia", "Omnia", "female", "ar", "Friendly, spontaneous Egyptian voice"),
]


class VoiceTutTTSBackend:
    """VoiceTut 0.6B Egyptian-Arabic TTS with 17 bundled studio speakers."""

    def __init__(self):
        self.model = None
        self.model_size = "default"
        self.device = self._get_device()
        self._resolved_model_path: Optional[str] = None

    def _get_device(self) -> str:
        device = get_torch_device(allow_mps=False, allow_xpu=False, allow_directml=False)
        return device if device.startswith("cuda") else "cpu"

    def _get_model_path(self, model_size: str = "default") -> str:
        return VOICETUT_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        override = os.environ.get("VOICETUT_MODEL_DIR")
        if override:
            root = Path(override)
            return (root / "config.json").exists() and (root / "model.safetensors").exists()
        return is_model_cached(
            VOICETUT_HF_REPO,
            required_files=["config.json", "model.safetensors"],
        )

    def is_loaded(self) -> bool:
        return self.model is not None

    async def load_model(self, model_size: str = "default") -> None:
        if self.model is not None:
            return
        await asyncio.to_thread(self._load_model_sync)

    def _resolve_model_snapshot(self) -> str:
        override = os.environ.get("VOICETUT_MODEL_DIR")
        if override:
            root = Path(override).resolve()
            if not (root / "model.safetensors").exists():
                raise FileNotFoundError(f"VoiceTut model.safetensors not found in {root}")
            return str(root)

        from huggingface_hub import snapshot_download

        return snapshot_download(
            VOICETUT_HF_REPO,
            allow_patterns=VOICETUT_MODEL_PATTERNS,
        )

    def _load_model_sync(self) -> None:
        from voicetut_tts import VoiceTutTTS

        cached = self._is_model_cached()
        with model_load_progress("voicetut", cached):
            model_path = self._resolve_model_snapshot()
            dtype = "float16" if self.device.startswith("cuda") else "float32"
            logger.info("Loading VoiceTut-TTS on %s (%s)...", self.device, dtype)
            self.model = VoiceTutTTS.from_pretrained(
                model_path,
                device=self.device,
                dtype=dtype,
                language="ar",
            )
            self._resolved_model_path = model_path

        logger.info("VoiceTut-TTS loaded with %d preset voices", len(VOICETUT_VOICES))

    def unload_model(self) -> None:
        if self.model is None:
            return
        del self.model
        self.model = None
        self._resolved_model_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VoiceTut-TTS unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        return {
            "voice_type": "preset",
            "preset_engine": "voicetut",
            "preset_voice_id": VOICETUT_DEFAULT_SPEAKER,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(
            audio_paths,
            reference_texts,
            sample_rate=VOICETUT_SAMPLE_RATE,
        )

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "ar",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        await self.load_model()
        speaker = voice_prompt.get("preset_voice_id") or VOICETUT_DEFAULT_SPEAKER

        def _generate_sync():
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            audio = self.model.synthesize(
                text,
                speaker=speaker,
                language="en" if language == "en" else "ar",
                normalize=True,
                num_step=32,
                guidance_scale=2.0,
            )
            return np.asarray(audio, dtype=np.float32), self.model.sampling_rate

        return await asyncio.to_thread(_generate_sync)
