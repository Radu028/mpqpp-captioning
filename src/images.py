import json
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from src.checks import sha256


def image_files(assets):
    digests = json.loads((assets / "image-sha256.json").read_text())
    drawbench = json.loads((assets / "drawbench.json").read_text())
    paths = {}

    for record in drawbench.values():
        for image in record["images"]:
            paths[str(image["image_id"])] = image["path"]

    files = {}
    for image_id, digest in digests.items():
        path = paths.get(image_id, f"coco/{int(image_id):012d}.jpg")
        files[path] = digest

    return files


def checked_path(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"Image path escapes the image folder or is a symlink: {path}")
    return path


def missing_images(root, expected):
    missing = []
    for relative, digest in expected.items():
        path = checked_path(root, relative)
        if not path.exists():
            missing.append(relative)
            continue
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Existing image differs, not overwriting: {path}")

    return missing


def download(release, target):
    url = (
        f"https://github.com/{release['repository']}/releases/download/"
        f"{release['tag']}/{release['asset']}"
    )
    print(f"Downloading {release['asset']}...", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def install_archive(archive, root, expected, release):
    if archive.stat().st_size != release["bytes"] or sha256(archive) != release["sha256"]:
        raise ValueError("Image ZIP has the wrong SHA256 or size. Nothing extracted.")

    missing = missing_images(root, expected)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = {f"images/{relative}" for relative in expected}
        archive_names = {member.filename for member in members}
        if len(members) != len(names) or archive_names != names:
            raise ValueError("Image archive has missing, duplicate or unexpected paths")

        for member in members:
            if stat.S_ISLNK(member.external_attr >> 16) or member.is_dir():
                raise ValueError("Image archive must contain only image files")

        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".image-extract-", dir=root) as temporary:
            staging = Path(temporary)
            # Unpack and check first, then move the images into place.
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
                    raise ValueError(f"File already exists, leaving it alone: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                (staging / relative).replace(destination)


def download_images(root, assets):
    root = Path(root)
    expected = image_files(assets)
    if not missing_images(root, expected):
        print(f"{len(expected)} images checked, nothing to download")
        return

    release = json.loads((assets / "image-release.json").read_text())
    root.mkdir(parents=True, exist_ok=True)

    # Download on the image disk so /tmp doesn't fill up.
    with tempfile.TemporaryDirectory(prefix=".image-download-", dir=root) as temporary:
        archive = Path(temporary) / release["asset"]
        download(release, archive)
        install_archive(archive, root, expected, release)
    print(f"{len(expected)} images saved and checked in {root}")
