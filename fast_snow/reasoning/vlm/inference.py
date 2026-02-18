"""VLM inference interface over 4DSG.

This module provides multiple backends for VLM inference:
1. Local: transformers pipeline or model (requires GPU)
2. Google AI Studio: Free API for Gemma/Gemini models
3. Hugging Face Inference API: Serverless inference
4. OpenAI-compatible API: For vLLM/TGI servers

Paper reference: SNOW uses Gemma3-4B-IT for spatial reasoning.
"""

from __future__ import annotations

import base64
import io
import os
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image as PILImage

from fast_snow.reasoning.graph.four_d_sg import FourDSceneGraph
from fast_snow.reasoning.vlm.prompt_builder import (
    PromptConfig,
    Phase7SerializationConfig,
    build_messages,
    build_simple_prompt,
    build_messages_phase7,
    build_prompt_phase7,
)

logger = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    """Configuration for VLM inference.

    Supported backends:
    - "pipeline": Local transformers pipeline (requires GPU)
    - "model": Local model loading (requires GPU)
    - "google_ai": Google AI Studio API (free, requires API key)
    - "huggingface": Hugging Face Inference API
    - "openai": OpenAI-compatible API (for vLLM/TGI servers)
    """
    model_id: str = "google/gemma-3-4b-it"
    device: str = "cuda"
    dtype: str = "bfloat16"
    backend: str = "pipeline"
    max_new_tokens: int = 256
    temperature: float = 0.0

    # API settings
    api_key: Optional[str] = None  # Will try env vars if None
    api_base_url: Optional[str] = None  # For OpenAI-compatible APIs

    # Prompt settings
    prompt_config: PromptConfig = field(default_factory=PromptConfig)


@dataclass
class Phase7VLMConfig:
    """Phase 7 strict VLM config (paper reproduction).

    These parameters are FIXED for Phase 7 to match paper specifications:
    - Model: Gemma3-4B-IT
    - Temperature: 0 (deterministic)
    - do_sample: false (enforced via temperature=0)
    - max_new_tokens: 256 (sufficient for multiple-choice answers)

    See docs/roadmap/SEMANTIC_SG_AND_SAM2_TEMPORAL_PLAN.md Phase 7 for details.
    """
    # Model (FIXED - paper specification)
    model_id: str = "google/gemma-3-4b-it"

    # Backend (configurable - depends on deployment)
    backend: str = "google_ai"  # Default: free Google AI Studio API

    # Inference parameters (FIXED - paper specification)
    temperature: float = 0.0  # Deterministic (do_sample=false enforced)
    max_new_tokens: int = 256  # Sufficient for answers

    # Hardware settings (configurable)
    device: str = "cuda"
    dtype: str = "bfloat16"

    # API settings (configurable)
    api_key: Optional[str] = None  # Will try env vars if None
    api_base_url: Optional[str] = None  # For OpenAI-compatible APIs

    # Serialization config (FIXED)
    serialization_config: Phase7SerializationConfig = field(
        default_factory=Phase7SerializationConfig
    )

    def __post_init__(self):
        """Validate that parameters match paper requirements."""
        if self.temperature != 0.0:
            logger.warning(
                f"Phase7VLMConfig: temperature={self.temperature} but paper requires 0.0. "
                f"Overriding to 0.0 for strict reproduction."
            )
            object.__setattr__(self, "temperature", 0.0)

        if self.max_new_tokens < 256:
            logger.warning(
                f"Phase7VLMConfig: max_new_tokens={self.max_new_tokens} but "
                f"recommended minimum is 256. Consider increasing."
            )


class VLMInterface:
    """Unified interface for VLM inference across multiple backends."""

    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None
        self._pipe = None
        self._client = None
        self._loaded = False

    def load(self) -> None:
        """Load the model or initialize API client."""
        if self._loaded:
            return

        backend = self.config.backend

        if backend == "pipeline":
            self._load_pipeline()
        elif backend == "model":
            self._load_model()
        elif backend == "google_ai":
            self._init_google_ai()
        elif backend == "huggingface":
            self._init_huggingface()
        elif backend == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self._loaded = True

    def _load_pipeline(self) -> None:
        """Load local transformers pipeline."""
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for local inference. "
                "Install with: pip install transformers torch"
            ) from exc

        device = 0 if self.config.device == "cuda" and torch.cuda.is_available() else -1
        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16

        logger.info(f"Loading pipeline for {self.config.model_id} on device {device}")
        self._pipe = pipeline(
            "text-generation",
            model=self.config.model_id,
            device=device,
            torch_dtype=dtype,
        )
        self._model = "pipeline"

    def _load_model(self) -> None:
        """Load local model directly."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for local inference."
            ) from exc

        logger.info(f"Loading model {self.config.model_id}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16,
        ).eval()
        self._processor = AutoTokenizer.from_pretrained(self.config.model_id)

    def _init_google_ai(self) -> None:
        """Initialize Google AI Studio client."""
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for Google AI backend. "
                "Install with: pip install google-genai"
            ) from exc

        api_key = self.config.api_key or os.environ.get("GOOGLE_AI_API_KEY")
        if not api_key:
            raise ValueError(
                "Google AI API key required. Set GOOGLE_AI_API_KEY env var or pass api_key in config."
            )

        # Map model names
        model_name = self.config.model_id
        if "gemma" in model_name.lower():
            model_name = "gemma-3-4b-it"  # or gemma-2-9b-it, etc.

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._model = "google_ai"
        logger.info(f"Initialized Google AI client for {model_name}")

    def _init_huggingface(self) -> None:
        """Initialize Hugging Face Inference API client."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for HF Inference API. "
                "Install with: pip install huggingface_hub"
            ) from exc

        api_key = self.config.api_key or os.environ.get("HF_TOKEN")

        self._client = InferenceClient(
            model=self.config.model_id,
            token=api_key,
        )
        self._model = "huggingface"
        logger.info(f"Initialized HF Inference client for {self.config.model_id}")

    def _init_openai(self) -> None:
        """Initialize OpenAI-compatible API client (for vLLM/TGI)."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAI-compatible API. "
                "Install with: pip install openai"
            ) from exc

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
        base_url = self.config.api_base_url or os.environ.get("OPENAI_API_BASE")

        if not base_url:
            raise ValueError(
                "api_base_url required for OpenAI-compatible backend. "
                "Set OPENAI_API_BASE env var or pass api_base_url in config."
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = "openai"
        logger.info(f"Initialized OpenAI-compatible client at {base_url}")

    def infer(self, query: str, graph: FourDSceneGraph) -> Dict[str, Any]:
        """Run inference on the 4DSG.

        Args:
            query: The question to answer about the scene.
            graph: 4D Scene Graph containing spatial-temporal information.

        Returns:
            Dict with query, model, answer, and metadata.
        """
        self.load()

        backend = self.config.backend

        if backend == "pipeline":
            answer = self._infer_pipeline(query, graph)
        elif backend == "model":
            answer = self._infer_model(query, graph)
        elif backend == "google_ai":
            answer = self._infer_google_ai(query, graph)
        elif backend == "huggingface":
            answer = self._infer_huggingface(query, graph)
        elif backend == "openai":
            answer = self._infer_openai(query, graph)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return {
            "query": query,
            "model": self.config.model_id,
            "backend": backend,
            "num_frames": len(graph.spatial_graphs),
            "num_tracks": len(graph.temporal_window.tracks),
            "answer": answer,
        }

    def _infer_pipeline(self, query: str, graph: FourDSceneGraph) -> str:
        """Inference using transformers pipeline."""
        prompt = build_simple_prompt(query, graph, self.config.prompt_config)

        output = self._pipe(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=self.config.temperature if self.config.temperature > 0 else None,
        )

        generated = output[0]["generated_text"]
        # Extract only the new generated text
        if generated.startswith(prompt):
            return generated[len(prompt):].strip()
        return generated.strip()

    def _infer_model(self, query: str, graph: FourDSceneGraph) -> str:
        """Inference using direct model loading."""
        import torch

        prompt = build_simple_prompt(query, graph, self.config.prompt_config)

        inputs = self._processor(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
            )

        generation = generation[0][input_len:]
        return self._processor.decode(generation, skip_special_tokens=True)

    def _infer_google_ai(self, query: str, graph: FourDSceneGraph) -> str:
        """Inference using Google AI Studio API."""
        from google.genai import types

        prompt = build_simple_prompt(query, graph, self.config.prompt_config)

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
        )

        return response.text

    def _infer_huggingface(self, query: str, graph: FourDSceneGraph) -> str:
        """Inference using Hugging Face Inference API."""
        prompt = build_simple_prompt(query, graph, self.config.prompt_config)

        response = self._client.text_generation(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature if self.config.temperature > 0 else None,
        )

        return response

    def _infer_openai(self, query: str, graph: FourDSceneGraph) -> str:
        """Inference using OpenAI-compatible API (vLLM/TGI)."""
        messages = build_messages(query, graph, self.config.prompt_config)

        # Convert to OpenAI format
        openai_messages = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                # Extract text from content blocks
                text = " ".join(
                    block["text"] for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                text = content
            openai_messages.append({"role": msg["role"], "content": text})

        response = self._client.chat.completions.create(
            model=self.config.model_id,
            messages=openai_messages,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )

        return response.choices[0].message.content

    def infer_text(self, prompt: str) -> str:
        """Direct text inference without 4DSG (for simple QA).

        Useful for VLM4D evaluation where we just need text QA.
        """
        self.load()

        backend = self.config.backend

        if backend == "pipeline":
            output = self._pipe(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
            )
            generated = output[0]["generated_text"]
            if generated.startswith(prompt):
                return generated[len(prompt):].strip()
            return generated.strip()

        elif backend == "google_ai":
            from google.genai import types
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )
            )
            return response.text

        elif backend == "huggingface":
            return self._client.text_generation(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
            )

        elif backend == "openai":
            response = self._client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def infer_multimodal(self, prompt: str, images: List[np.ndarray]) -> str:
        """Multimodal inference with text prompt and images.

        Sends both text and images to the VLM for vision-language reasoning.
        Currently supported backends: google_ai, openai.
        Other backends fall back to text-only inference with a warning.

        Args:
            prompt: The text prompt to send alongside the images.
            images: List of images as numpy arrays (H, W, C) in uint8 RGB format.

        Returns:
            The model's text response.
        """
        self.load()

        backend = self.config.backend

        if backend == "google_ai":
            # Convert numpy images to PIL
            pil_images = [PILImage.fromarray(img) for img in images]

            # Build content parts: text first, then images
            parts: List[Any] = [prompt]
            for pil_img in pil_images:
                parts.append(pil_img)

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=parts,
                config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_new_tokens,
                },
            )
            return response.text

        elif backend == "openai":
            # Encode images as base64 JPEG for the OpenAI vision API
            content_parts: List[Dict[str, Any]] = [
                {"type": "text", "text": prompt}
            ]
            for img in images:
                pil_img = PILImage.fromarray(img)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

            response = self._client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": content_parts}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
            )
            return response.choices[0].message.content

        else:
            # Fallback for unsupported backends
            warnings.warn(
                f"Backend '{backend}' does not support multimodal. "
                f"Falling back to text-only."
            )
            return self.infer_text(prompt)


def create_vlm_interface(
    backend: str = "google_ai",
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> VLMInterface:
    """Convenience function to create a VLM interface.

    Args:
        backend: One of "pipeline", "model", "google_ai", "huggingface", "openai"
        model_id: Model identifier (defaults based on backend)
        api_key: API key (will try env vars if not provided)
        api_base_url: Base URL for OpenAI-compatible APIs

    Returns:
        Configured VLMInterface ready for inference.

    Example:
        # Using Google AI Studio (free)
        vlm = create_vlm_interface(backend="google_ai")
        result = vlm.infer("How many cars?", scene_graph)

        # Using local vLLM server
        vlm = create_vlm_interface(
            backend="openai",
            api_base_url="http://localhost:8000/v1",
            model_id="google/gemma-3-4b-it"
        )
    """
    # Default model IDs per backend
    default_models = {
        "pipeline": "google/gemma-3-4b-it",
        "model": "google/gemma-3-4b-it",
        "google_ai": "gemma-3-4b-it",
        "huggingface": "google/gemma-3-4b-it",
        "openai": "gemma-3-4b-it",
    }

    if model_id is None:
        model_id = default_models.get(backend, "google/gemma-3-4b-it")

    config = VLMConfig(
        model_id=model_id,
        backend=backend,
        api_key=api_key,
        api_base_url=api_base_url,
    )

    return VLMInterface(config)


def create_phase7_vlm_interface(
    backend: str = "google_ai",
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> VLMInterface:
    """Create VLM interface with Phase 7 strict parameters.

    This is the RECOMMENDED interface for Phase 7 paper reproduction.
    All parameters are fixed to match paper specifications:
    - Model: Gemma3-4B-IT
    - Temperature: 0 (deterministic)
    - max_new_tokens: 256

    Args:
        backend: One of "pipeline", "model", "google_ai", "huggingface", "openai"
        api_key: API key (will try env vars if not provided)
        api_base_url: Base URL for OpenAI-compatible APIs

    Returns:
        VLMInterface with Phase 7 strict config

    Example:
        # Using Google AI Studio (recommended for Phase 7)
        vlm = create_phase7_vlm_interface(backend="google_ai")
        result = vlm.infer("How many cars?", scene_graph)
    """
    # Create Phase 7 config (parameters are fixed)
    phase7_config = Phase7VLMConfig(
        backend=backend,
        api_key=api_key,
        api_base_url=api_base_url,
    )

    # Convert to VLMConfig for VLMInterface
    # (VLMInterface doesn't know about Phase7VLMConfig yet)
    vlm_config = VLMConfig(
        model_id=phase7_config.model_id,
        backend=phase7_config.backend,
        device=phase7_config.device,
        dtype=phase7_config.dtype,
        max_new_tokens=phase7_config.max_new_tokens,
        temperature=phase7_config.temperature,
        api_key=phase7_config.api_key,
        api_base_url=phase7_config.api_base_url,
    )

    return VLMInterface(vlm_config)


def infer_phase7(
    query: str,
    graph: FourDSceneGraph,
    vlm: VLMInterface,
    config: Optional[Phase7SerializationConfig] = None,
) -> str:
    """Phase 7 VLM inference (strict version).

    This is the UNIFIED entry point for Phase 7 VLM inference.
    It ensures:
    1. Deterministic 4DSG serialization (strict JSON)
    2. Fixed VLM parameters (temperature=0, do_sample=false)
    3. Proper prompt formatting

    Args:
        query: User question
        graph: 4D Scene Graph
        vlm: VLM interface (should use create_phase7_vlm_interface())
        config: Phase 7 serialization config

    Returns:
        VLM answer (text)

    Example:
        vlm = create_phase7_vlm_interface(backend="google_ai")
        answer = infer_phase7("How many cars?", scene_graph, vlm)
    """
    if config is None:
        config = Phase7SerializationConfig()

    # Build messages using Phase 7 strict format
    messages = build_messages_phase7(query, graph, config)

    # For backends that support chat format
    backend = vlm.config.backend
    if backend in ["google_ai", "openai"]:
        # Use chat format
        # Extract text from messages
        system_text = messages[0]["content"][0]["text"]
        user_text = messages[1]["content"][0]["text"]

        full_prompt = f"{system_text}\n\n{user_text}"

        return vlm.infer_text(full_prompt)
    else:
        # Use simple prompt for other backends
        prompt = build_prompt_phase7(query, graph, config)
        return vlm.infer_text(prompt)
