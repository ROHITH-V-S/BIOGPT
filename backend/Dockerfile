# Stage 1: Builder
FROM python:3.11-slim AS builder

# A venv rather than pip --prefix: with --prefix the second pip invocation
# cannot see what the first installed, so it re-resolves torch and pulls the
# CUDA build anyway. Inside a venv "already satisfied" works as expected.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .

# CPU-only torch, installed first and deliberately.
#
# sentence-transformers depends on torch, and the default PyPI wheel bundles the
# full CUDA toolchain: 2.7 GB of nvidia/* plus 691 MB of triton. None of it can
# ever execute here — the container is not built with a GPU runtime — so it was
# ~3.4 GB of an image that could not use it. Resolving torch from PyTorch's CPU
# index first means the later requirements install finds it already satisfied.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app

# Create non-root user
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser

# Copy the prepared environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy app code
COPY . .

# Install the biomedical NER model.
#
# This runs in the runtime stage, not the builder, because the script resolves
# its target with site.getsitepackages() — it must run against the same
# interpreter that will later import the model.
#
# Deliberately allowed to fail the build. Without the model app/ner.py silently
# falls back to regex extraction, which would ship an image whose headline
# feature is quietly degraded — a louder failure is the safer one.
RUN python scripts/install_ner_model.py

# Create data directory
RUN mkdir -p data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# --proxy-headers makes uvicorn read X-Forwarded-For, so request.client.host is
# the real client rather than the nginx container. Without it the per-IP rate
# limiter sees one address for every visitor and they share a single bucket.
#
# --forwarded-allow-ips="*" trusts those headers from any peer. That is fine
# while only the reverse proxy can reach this port; if port 8000 is exposed to
# untrusted networks, narrow it to the proxy's address, since a direct caller
# could otherwise spoof X-Forwarded-For to dodge rate limiting.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
