"""
BioGPT Explorer backend package.

OpenMP coexistence (Windows)
----------------------------
spaCy's thinc/blis stack loads Intel's OpenMP runtime (``libiomp5md.dll``) at
import time, and faiss lazily initializes LLVM's (``libomp140.x86_64.dll``) on
its first index search. The second runtime to initialize aborts the process
with "OMP: Error #15", which means enabling biomedical NER would otherwise
crash every retrieval call.

Allowing both to coexist is the supported-in-practice fix for this pairing.
It must be set before either library is imported, so it lives here — this
package's __init__ runs ahead of every ``from app...`` import.
"""

import os
import sys

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
