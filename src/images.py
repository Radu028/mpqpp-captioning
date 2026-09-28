"""Download the fixed image set and verify it before installing missing files."""

import json
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from src.checks import sha256


def image_files(assets):
    digests = json.loads((assets / "image-sha256.json").read_text())
    drawbench = json.loads((assets / "drawbench.json").read_text())
    paths = {str(image["image_id"]): image["path"]
             for record in drawbench.values() for image in record["images"]}
    return {paths.get(image_id, f"coco/{int(image_id):012d}.jpg"): digest
            for image_id, digest in digests.items()}


def checked_path(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"Image path escapes the image folder or is a symlink: {path}")
    return path


def missing_images(root, expected):
    missing = []
    for relative, digest in expected.items():
        path = checked_path(root, relative)
        if path.exists():
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"Existing image differs; not overwriting: {path}")
        else:
            missing.append(relative)
    return missing


def download(release, target):
    url = (f"https://github.com/{release['repository']}/releases/download/"
           f"{release['tag']}/{release['asset']}")
    print(f"Downloading {release['asset']} from GitHub Releases...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as error:
        raise ValueError(f"Cannot download the public image release: HTTP {error.code}. "
                         f"Check {url} and retry.") from error


def install_archive(archive, root, expected, release):
    if archive.stat().st_size != release["bytes"] or sha256(archive) != release["sha256"]:
        raise ValueError("Image archive SHA256 or size mismatch; no images installed")
    missing = missing_images(root, expected)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = {f"images/{relative}" for relative in expected}
        if len(members) != len(names) or {m.filename for m in members} != names:
            raise ValueError("Image archive has missing, duplicate or unexpected paths")
        if any(stat.S_ISLNK(m.external_attr >> 16) or m.is_dir() for m in members):
            raise ValueError("Image archive must contain only image files")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".image-extract-", dir=root) as temporary:
            staging = Path(temporary)
            # Check every member, including files already present, before publishing any.
            for relative, digest in expected.items():
                path = checked_path(staging, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(f"images/{relative}") as source, path.open("wb") as out:
                    shutil.copyfileobj(source, out)
                if sha256(path) != digest:
                    raise ValueError(f"Image SHA256 mismatch: {relative}")
            for relative in missing:
                destination = checked_path(root, relative)
                if destination.exists():
                    raise ValueError(f"Image appeared during extraction; not overwriting: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                (staging / relative).replace(destination)


def download_images(root, assets):
    root = Path(root)
    expected = image_files(assets)
    if not missing_images(root, expected):
        print(f"All {len(expected)} images are present and SHA256-verified; no download needed")
        return
    release = json.loads((assets / "image-release.json").read_text())
    root.mkdir(parents=True, exist_ok=True)
    # Keep the temporary archive on the image disk, not the system temporary disk.
    with tempfile.TemporaryDirectory(prefix=".image-download-", dir=root) as temporary:
        archive = Path(temporary) / release["asset"]
        download(release, archive)
        install_archive(archive, root, expected, release)
    print(f"Installed and SHA256-verified all {len(expected)} images in {root}")
