"""
Install the scispaCy biomedical NER model (``en_core_sci_sm``).

Why this exists instead of a line in requirements.txt
-----------------------------------------------------
``pip install <scispacy model url>`` fails on modern Python: the model's
setup.py needs Cython at build time, and scispaCy itself pins ``nmslib``,
which has no wheels for 3.12+. The model payload, however, is just a data
directory — no compilation is actually required.

This script therefore:

1. downloads the model tarball,
2. copies the importable package into site-packages, and
3. rewrites the quoted booleans in ``config.cfg`` (``"False"`` → ``false``),
   which spaCy 3.7 tolerated but 3.8+ rejects with a ConfigValidationError.

Run once after creating the venv::

    python scripts/install_ner_model.py

Without it the app still runs — ``app/ner.py`` falls back to regex extraction —
but entity-aware retrieval operates on much cruder entities.
"""

import re
import shutil
import site
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

MODEL = "en_core_sci_sm"
VERSION = "0.5.4"
URL = (
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/"
    f"v{VERSION}/{MODEL}-{VERSION}.tar.gz"
)

# spaCy >= 3.8 validates config types strictly; these were emitted as strings.
QUOTED_BOOL = re.compile(r'=\s*"(True|False)"')


def _site_packages() -> Path:
    paths = site.getsitepackages()
    for p in paths:
        if p.endswith("site-packages"):
            return Path(p)
    return Path(paths[-1])


def _patch_config(model_dir: Path) -> int:
    """Rewrite ``key = "False"`` as ``key = false`` in every config.cfg."""
    patched = 0
    for cfg in model_dir.rglob("config.cfg"):
        text = cfg.read_text(encoding="utf-8")
        new = QUOTED_BOOL.sub(lambda m: f"= {m.group(1).lower()}", text)
        if new != text:
            cfg.write_text(new, encoding="utf-8")
            patched += 1
    return patched


def main() -> int:
    sp = _site_packages()
    print(f"site-packages: {sp}")

    tmp = Path(tempfile.mkdtemp())
    archive = tmp / f"{MODEL}.tar.gz"

    print(f"downloading {URL} …")
    urllib.request.urlretrieve(URL, archive)
    print(f"  {archive.stat().st_size:,} bytes")

    with tarfile.open(archive) as tf:
        tf.extractall(tmp)

    candidates = [p.parent for p in tmp.rglob("__init__.py") if p.parent.name == MODEL]
    if not candidates:
        print(f"ERROR: no importable '{MODEL}' package inside the tarball")
        return 1

    dst = sp / MODEL
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(candidates[0], dst)
    print(f"installed  -> {dst}")

    n = _patch_config(dst)
    print(f"patched    -> {n} config.cfg file(s) for spaCy 3.8+ strict validation")

    shutil.rmtree(tmp, ignore_errors=True)

    # Verify it actually loads through the app's own code path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.ner import SPACY_AVAILABLE, extract_entities  # noqa: E402

    if not SPACY_AVAILABLE:
        print("\nWARNING: model installed but app/ner.py still fell back to regex.")
        return 1

    demo = "BRCA1 and BRCA2 mutations increase breast cancer risk."
    print(f"\nscispaCy active. Sample extraction:\n  {demo}\n  {extract_entities(demo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
