import hashlib
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.checks import sha256
from src.images import download_images, install_archive


class ImageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "images"
        self.archive = self.directory / "images.zip"
        self.relative = "coco/000000000001.jpg"
        self.contents = b"image bytes"
        self.expected = {self.relative: hashlib.sha256(self.contents).hexdigest()}
        self.release = self.make_archive({f"images/{self.relative}": self.contents})

    def make_archive(self, entries):
        with zipfile.ZipFile(self.archive, "w") as bundle:
            for name, contents in entries.items():
                bundle.writestr(name, contents)
        return {
            "asset": "images.zip",
            "bytes": self.archive.stat().st_size,
            "sha256": sha256(self.archive),
        }

    def test_download_once(self):
        assets = self.directory / "assets"
        assets.mkdir()
        (assets / "image-sha256.json").write_text(json.dumps({"1": self.expected[self.relative]}))
        (assets / "drawbench.json").write_text("{}")
        (assets / "image-release.json").write_text(json.dumps(self.release))
        with patch(
            "src.images.download",
            side_effect=lambda _, target: shutil.copyfile(self.archive, target),
        ) as fetch:
            download_images(self.root, assets)
            self.assertEqual((self.root / self.relative).read_bytes(), self.contents)
            modified = (self.root / self.relative).stat().st_mtime_ns
            download_images(self.root, assets)
            fetch.assert_called_once()
            self.assertEqual((self.root / self.relative).stat().st_mtime_ns, modified)
        self.assertFalse(list(self.root.glob(".image-*")))

    def test_bad_zip_hash(self):
        with self.assertRaisesRegex(ValueError, "wrong SHA256"):
            install_archive(
                self.archive, self.root, self.expected, dict(self.release, sha256="wrong")
            )
        self.assertFalse(self.root.exists())

    def test_bad_image_hash(self):
        release = self.make_archive({f"images/{self.relative}": b"wrong image"})
        with self.assertRaisesRegex(ValueError, "Image SHA256"):
            install_archive(self.archive, self.root, self.expected, release)
        self.assertFalse((self.root / self.relative).exists())

    def test_path_escape(self):
        release = self.make_archive({"images/../../outside.jpg": self.contents})
        with self.assertRaisesRegex(ValueError, "unexpected paths"):
            install_archive(self.archive, self.root, self.expected, release)
        self.assertFalse(self.root.exists())

    def test_keep_existing_image(self):
        path = self.root / self.relative
        path.parent.mkdir(parents=True)
        path.write_bytes(b"user image")
        with self.assertRaisesRegex(ValueError, "not overwriting"):
            install_archive(self.archive, self.root, self.expected, self.release)
        self.assertEqual(path.read_bytes(), b"user image")

    def test_symlink_escape(self):
        outside = self.directory / "outside"
        outside.mkdir()
        self.root.mkdir()
        (self.root / "coco").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            install_archive(self.archive, self.root, self.expected, self.release)
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
