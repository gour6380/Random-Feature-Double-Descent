"""Standard-library regressions for byte-stable public result manifests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from check_repository import validate_result_manifest


class ResultManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package = self.root / "results"
        self.package.mkdir()
        self.manifest = self.package / "manifest.json"

    def write_manifest(self, entries: dict[str, bytes]) -> None:
        self.manifest.write_text(
            json.dumps(
                {
                    "files": {
                        name: hashlib.sha256(content).hexdigest()
                        for name, content in entries.items()
                    }
                }
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def write_artifact(self, relative: str, content: bytes) -> None:
        path = self.package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def test_lf_utf8_text_passes_with_matching_raw_hash(self) -> None:
        content = "label,value\nσ,0.3\n".encode("utf-8")
        self.write_artifact("tables/measurements.csv", content)
        self.write_manifest({"tables/measurements.csv": content})

        errors, count = validate_result_manifest(self.manifest)

        self.assertEqual(errors, [])
        self.assertEqual(count, 1)

    def test_matching_hash_crlf_text_is_rejected_before_normalization(self) -> None:
        content = b"label,value\r\nnoise,0.3\r\n"
        self.write_artifact("tables/measurements.csv", content)
        self.write_manifest({"tables/measurements.csv": content})

        errors, count = validate_result_manifest(self.manifest)

        self.assertTrue(any("CRLF" in error for error in errors), errors)
        self.assertEqual(count, 1)

    def test_lf_normalization_does_not_allow_a_stale_raw_hash(self) -> None:
        original = b"label,value\r\nnoise,0.3\r\n"
        self.write_artifact("tables/measurements.csv", original.replace(b"\r\n", b"\n"))
        self.write_manifest({"tables/measurements.csv": original})

        errors, count = validate_result_manifest(self.manifest)

        self.assertTrue(any("checksum" in error.lower() for error in errors), errors)
        self.assertEqual(count, 1)

    def test_changed_measurement_is_rejected(self) -> None:
        original = b"label,value\nnoise,0.3\n"
        self.write_artifact("tables/measurements.csv", original.replace(b"0.3", b"0.4"))
        self.write_manifest({"tables/measurements.csv": original})

        errors, count = validate_result_manifest(self.manifest)

        self.assertTrue(any("checksum" in error.lower() for error in errors), errors)
        self.assertEqual(count, 1)

    def test_binary_crlf_bytes_remain_valid(self) -> None:
        # PNG signatures contain CRLF bytes; the manifest check is not an image decoder.
        content = b"\x89PNG\r\n\x1a\nsynthetic-binary-fixture\r\n"
        self.write_artifact("figures/preview.png", content)
        self.write_manifest({"figures/preview.png": content})

        errors, count = validate_result_manifest(self.manifest)

        self.assertEqual(errors, [])
        self.assertEqual(count, 1)

    def test_path_outside_package_is_rejected_even_with_matching_hash(self) -> None:
        content = b"outside package\n"
        (self.root / "outside.csv").write_bytes(content)
        self.write_manifest({"../outside.csv": content})

        errors, count = validate_result_manifest(self.manifest)

        self.assertTrue(any("escape" in error.lower() for error in errors), errors)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
