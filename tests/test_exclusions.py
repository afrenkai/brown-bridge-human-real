import json
from pathlib import Path

from brown_bridge.exclusions import filter_excluded_paths, sha256_file
from brown_bridge.features import compact_spatial_content_descriptor
import torch


def test_filter_excludes_exact_hash_but_not_same_name_with_different_content(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a" / "sample.png"
    second = tmp_path / "b" / "sample.png"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"excluded")
    second.write_bytes(b"retained")
    manifest = tmp_path / "exclusions.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": first.name,
                        "path": str(first),
                        "sha256": sha256_file(first),
                    }
                ]
            }
        )
    )
    assert filter_excluded_paths([first, second], manifest) == [second]


def test_compact_content_descriptor_keeps_coarse_spatial_layout() -> None:
    global_features = torch.randn(3, 8)
    patch_features = torch.randn(3, 16, 8)
    descriptor = compact_spatial_content_descriptor(
        global_features, patch_features, torch.randn(8)
    )
    assert descriptor.shape == (3, 8 + 4 * 8)
    assert torch.isfinite(descriptor).all()
