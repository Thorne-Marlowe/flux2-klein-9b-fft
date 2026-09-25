# syntax=docker/dockerfile:1
#
# Lean Linux amd64 runtime for the Klein Base 9B trainer. The base image is
# deliberately supplied as a build argument so release builds can pin it to a
# digest (for example, python:3.12.3-slim-bookworm@sha256:...).

ARG PYTHON_BASE_IMAGE=python:3.12.3-slim-bookworm
FROM --platform=linux/amd64 ${PYTHON_BASE_IMAGE}

ARG SOURCE_COMMIT=unknown
ARG ENVIRONMENT_VERSION=unversioned
ARG REQUIREMENTS_LOCK_SHA256=unknown

LABEL org.opencontainers.image.source="https://github.com/Thorne-Marlowe/flux2-klein-9b-fft" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.version="${ENVIRONMENT_VERSION}" \
      org.opencontainers.image.title="FLUX.2 Klein Base 9B trainer environment" \
      org.opencontainers.image.description="Pinned runtime for the standalone Klein Base 9B trainer; models and run state stay outside the image." \
      org.opencontainers.image.licenses="MIT" \
      com.thorne-marlowe.klein.requirements-sha256="${REQUIREMENTS_LOCK_SHA256}"

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgomp is a small runtime dependency used by the prebuilt numerical wheels.
# No compiler, CUDA toolkit, or second CUDA package source is added here; CUDA
# and PyTorch come from requirements-smoke.txt.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv "${VIRTUAL_ENV}"

WORKDIR /opt/flux2-klein-9b-fft

# Install the repository's existing lock before copying source so dependency
# layers remain reusable when only trainer files change.
COPY requirements-smoke.txt /tmp/requirements-smoke.txt
RUN python -m pip install --disable-pip-version-check --no-cache-dir \
        --only-binary=:all: -r /tmp/requirements-smoke.txt \
    && python -m pip check \
    && python -c "import torch, torchvision, diffusers, transformers, accelerate, safetensors; assert torch.version.cuda == '12.8', torch.version.cuda"

COPY . /opt/flux2-klein-9b-fft

# Build metadata is intentionally small and contains no environment or
# credential values. The lock digest is computed from the file in the image;
# the build argument is retained in the OCI label for registry inspection.
RUN LOCK_SHA256="$(sha256sum requirements-smoke.txt | cut -d ' ' -f1)" \
    && printf '{"source_commit":"%s","environment_version":"%s","requirements_lock_sha256":"%s","repository":"%s"}\n' \
       "${SOURCE_COMMIT}" "${ENVIRONMENT_VERSION}" "${LOCK_SHA256}" \
       "/opt/flux2-klein-9b-fft" > /opt/flux2-klein-9b-fft/.container-build-info.json \
    && chmod 0644 /opt/flux2-klein-9b-fft/.container-build-info.json

COPY docker/entrypoint.sh /usr/local/bin/klein-entrypoint
RUN chmod 0755 /usr/local/bin/klein-entrypoint

ENTRYPOINT ["/usr/local/bin/klein-entrypoint"]
