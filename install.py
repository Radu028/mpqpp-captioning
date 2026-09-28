"""Creates .venv-blip2, .venv-centurio and .venv-eval (the models need different transformers versions)."""

import subprocess
import venv

from run import ROOT, venv_python

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

for name in ("blip2", "centurio", "eval"):
    print(f"== .venv-{name}", flush=True)
    if not venv_python(name).exists():
        venv.create(ROOT / f".venv-{name}", with_pip=True)
    pip = [str(venv_python(name)), "-m", "pip", "install"]
    subprocess.run(pip + ["-r", ROOT / "requirements" / "torch.txt", "--index-url", TORCH_INDEX], check=True)
    subprocess.run(pip + ["-r", ROOT / "requirements" / f"{name}.txt"], check=True)
