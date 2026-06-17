import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

DIFFICULTY_ORDER = ("hard", "medium", "easy")

# Ratios used when first generating a manifest if one does not exist yet.
DEFAULT_BUCKET_SPLIT: Dict[str, float] = {
    "hard": 0.2,
    "medium": 0.35,
    "easy": 0.45,
}

# Ratios used when selecting a subset for a training run.
DEFAULT_SELECTION_MIX: Dict[str, float] = {
    "hard": 0.25,
    "medium": 0.35,
    "easy": 0.40,
}


def _normalize_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    normalized = {level: float(ratios.get(level, 0.0)) for level in DIFFICULTY_ORDER}
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("At least one difficulty bucket must have a positive weight.")
    return {level: normalized[level] / total for level in DIFFICULTY_ORDER}


def _ordered_image_ids(
    base_coco: Dict,
    ranked_ids: Iterable[int],
) -> List[int]:
    image_lookup = {img["id"] for img in base_coco.get("images", [])}
    ordered_ids: List[int] = []
    seen = set()

    for image_id in ranked_ids:
        if image_id in image_lookup and image_id not in seen:
            ordered_ids.append(image_id)
            seen.add(image_id)

    for image in sorted(base_coco.get("images", []), key=lambda item: item["id"]):
        image_id = image["id"]
        if image_id not in seen:
            ordered_ids.append(image_id)
            seen.add(image_id)

    return ordered_ids


def build_manifest(
    base_coco: Dict,
    ranked_ids: Iterable[int],
    dataset_name: str,
    bucket_split: Optional[Dict[str, float]] = None,
) -> Dict:
    split = _normalize_ratios(bucket_split or DEFAULT_BUCKET_SPLIT)
    ordered_ids = _ordered_image_ids(base_coco, ranked_ids)
    total_images = len(ordered_ids)

    hard_cut = int(round(total_images * split["hard"]))
    medium_cut = int(round(total_images * split["medium"]))
    if hard_cut + medium_cut > total_images:
        overflow = hard_cut + medium_cut - total_images
        medium_cut = max(0, medium_cut - overflow)

    hard_ids = ordered_ids[:hard_cut]
    medium_ids = ordered_ids[hard_cut : hard_cut + medium_cut]
    easy_ids = ordered_ids[hard_cut + medium_cut :]

    entries = (
        [{"image_id": image_id, "difficulty": "hard"} for image_id in hard_ids]
        + [{"image_id": image_id, "difficulty": "medium"} for image_id in medium_ids]
        + [{"image_id": image_id, "difficulty": "easy"} for image_id in easy_ids]
    )

    return {
        "dataset": dataset_name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "bucket_split": split,
        "entries": entries,
    }


def _write_manifest(path: str, manifest: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=4)


def build_or_load_manifest(
    manifest_path: str,
    base_coco: Dict,
    ranked_ids: Iterable[int],
    dataset_name: str,
    bucket_split: Optional[Dict[str, float]] = None,
) -> Dict:
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        entries = manifest.get("entries")
        if entries:
            return manifest

    manifest = build_manifest(base_coco, ranked_ids, dataset_name, bucket_split=bucket_split)
    _write_manifest(manifest_path, manifest)
    return manifest


def _allocate_counts(
    requested_total: int,
    mix: Dict[str, float],
    available: Dict[str, int],
) -> Tuple[Dict[str, int], int]:
    total_available = sum(available.values())
    total_to_allocate = min(requested_total, total_available)

    counts = {level: 0 for level in DIFFICULTY_ORDER}
    if total_to_allocate <= 0:
        return counts, 0

    desired = {level: total_to_allocate * mix[level] for level in DIFFICULTY_ORDER}
    allocated = 0

    for level in DIFFICULTY_ORDER:
        take = min(int(desired[level]), available[level])
        counts[level] = take
        allocated += take

    remainder = total_to_allocate - allocated
    fractional_order = sorted(
        DIFFICULTY_ORDER,
        key=lambda level: (desired[level] - int(desired[level]), mix[level]),
        reverse=True,
    )

    while remainder > 0:
        progressed = False
        for level in fractional_order:
            if counts[level] < available[level]:
                counts[level] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
        if not progressed:
            break

    actual_total = total_to_allocate - remainder
    return counts, actual_total


def select_anchor_subset(
    manifest: Dict,
    requested_total: int,
    mix: Optional[Dict[str, float]] = None,
) -> Tuple[List[int], Dict[str, int], int]:
    buckets: Dict[str, List[int]] = {level: [] for level in DIFFICULTY_ORDER}
    for entry in manifest.get("entries", []):
        difficulty = entry.get("difficulty")
        image_id = entry.get("image_id")
        if difficulty in buckets:
            buckets[difficulty].append(image_id)

    available = {level: len(buckets[level]) for level in DIFFICULTY_ORDER}
    mix_to_use = _normalize_ratios(mix or manifest.get("selection_mix") or DEFAULT_SELECTION_MIX)
    counts, actual_total = _allocate_counts(requested_total, mix_to_use, available)

    selected: List[int] = []
    for level in DIFFICULTY_ORDER:
        selected.extend(buckets[level][: counts[level]])

    return selected, counts, actual_total


__all__ = [
    "DEFAULT_BUCKET_SPLIT",
    "DEFAULT_SELECTION_MIX",
    "DIFFICULTY_ORDER",
    "build_manifest",
    "build_or_load_manifest",
    "select_anchor_subset",
]
