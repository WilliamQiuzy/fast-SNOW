"""VLM4D dataset loader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse
import urllib.request
import json


@dataclass(frozen=True)
class VLM4DSample:
    sample_id: str
    video: str
    question_type: str
    question: str
    choices: Dict[str, str]
    answer: str


def load_vlm4d_json(json_path: Path) -> List[VLM4DSample]:
    """Load a VLM4D JSON file into a list of samples."""
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    samples: List[VLM4DSample] = []
    for item in raw:
        samples.append(
            VLM4DSample(
                sample_id=item["id"],
                video=item["video"],
                question_type=item["question_type"],
                question=item["question"],
                choices=item["choices"],
                answer=item["answer"],
            )
        )
    return samples


def _resolve_video_from_hf_url(video_url: str, local_root: Path) -> Optional[Path]:
    """Resolve a HuggingFace URL to a local path under local_root.

    Example URL path suffix:
    /datasets/shijiezhou/VLM4D/resolve/main/videos_real/davis/city-ride.mp4
    """
    parsed = urlparse(video_url)
    if not parsed.path:
        return None
    parts = parsed.path.split("/resolve/main/")
    if len(parts) != 2:
        return None
    rel = parts[1].lstrip("/")
    candidate = local_root / rel
    if candidate.exists():
        return candidate
    # Fallback: search by basename inside local_root
    basename = Path(rel).name
    for path in local_root.rglob(basename):
        return path
    return None


def download_video(video_url: str, local_video_root: Path) -> Optional[Path]:
    """Download a video URL into local_video_root, preserving HF path suffix."""
    parsed = urlparse(video_url)
    if not parsed.path:
        return None
    parts = parsed.path.split("/resolve/main/")
    if len(parts) == 2:
        rel = parts[1].lstrip("/")
        target = local_video_root / rel
    else:
        target = local_video_root / "_misc" / Path(parsed.path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(video_url, target)
    return target


def resolve_video_path(
    video_url: str,
    local_video_root: Optional[Path] = None,
    url_to_local: Optional[Dict[str, Path]] = None,
    allow_download: bool = False,
) -> str:
    """Resolve a video reference to a local path when available.

    Returns a string path if resolved locally; otherwise returns the original URL.
    """
    if url_to_local and video_url in url_to_local:
        return str(url_to_local[video_url])
    if local_video_root:
        resolved = _resolve_video_from_hf_url(video_url, local_video_root)
        if resolved is not None:
            return str(resolved)
        if allow_download:
            downloaded = download_video(video_url, local_video_root)
            if downloaded is not None:
                return str(downloaded)
    return video_url


def iter_vlm4d_samples(
    json_path: Path,
    local_video_root: Optional[Path] = None,
    url_to_local: Optional[Dict[str, Path]] = None,
    question_type: Optional[str] = None,
) -> Iterable[VLM4DSample]:
    """Yield VLM4D samples with resolved video paths."""
    for sample in load_vlm4d_json(json_path):
        if question_type and sample.question_type != question_type:
            continue
        resolved_video = resolve_video_path(sample.video, local_video_root, url_to_local)
        yield VLM4DSample(
            sample_id=sample.sample_id,
            video=resolved_video,
            question_type=sample.question_type,
            question=sample.question,
            choices=sample.choices,
            answer=sample.answer,
        )
