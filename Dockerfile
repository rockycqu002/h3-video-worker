# syntax=docker/dockerfile:1.7
# MiniMax H3 (DaSiWa Hybrid 8-step) video worker for RunPod Serverless on RTX 4090 (CUDA 13 / driver >= 580).
# Environment layout follows ~/qwen-image-edit-4090 (validated on RunPod 4090s): digest-pinned CUDA 13 base, torch 2.14+cu130,
# ComfyUI pinned by commit, pip pins from constraints.txt. Unlike that image the ~42 GB of weights are NOT baked in:
# they live on a RunPod Network Volume at /runpod-volume/h3/models (see scripts/fetch_models.py).
FROM nvidia/cuda:13.0.1-base-ubuntu24.04@sha256:f8ef28f579ea42a44b415d2c5d46f788e6a9b395c6c83f2929416e1fc192c143 AS runtime

# ComfyUI 0.37.0 — has the native MiniMaxH3* nodes and --disable-nvml-pressure
ARG COMFY_COMMIT=88ab4a06566454ad89db8f0bedb970d6c08cd1b7
# open-h3-ir 0.4.1 (main @ 2026-08-23), the version validated on AutoDL
ARG H3IR_COMMIT=fd031e136ae3d89147324c0d1b2aa65e838c21f5

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH HF_HUB_DISABLE_TELEMETRY=1 COMFY_DIR=/app/ComfyUI RUNPOD_LOG_LEVEL=WARN

# gcc + libc6-dev + python3.12-dev: Triton JIT-compiles kernels at runtime and needs a C compiler and Python.h
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev gcc libc6-dev git tini ca-certificates curl libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv && pip install "pip==25.3"

WORKDIR /app
COPY constraints.txt ./

# torch 2.14.0 on PyPI is the CUDA 13.0 build
RUN pip install -c constraints.txt torch==2.14.0 torchvision==0.29.0 torchaudio==2.11.0

# ComfyUI at the pinned commit, without git history; its optional template gallery package is skipped
RUN git init -q ComfyUI && cd ComfyUI \
    && git remote add origin https://github.com/Comfy-Org/ComfyUI.git \
    && git fetch -q --depth 1 origin ${COMFY_COMMIT} && git checkout -q FETCH_HEAD && rm -rf .git \
    && grep -v '^comfyui-workflow-templates' requirements.txt > /tmp/comfy-req.txt \
    && pip install -c /app/constraints.txt -r /tmp/comfy-req.txt \
    && grep -q '"MiniMaxH3ImageToVideo"' comfy_extras/nodes_minimax_h3.py && grep -q '"MiniMaxH3SigmaShift"' comfy_extras/nodes_minimax_h3.py

COPY requirements.txt ./
RUN pip install -c constraints.txt -r requirements.txt

# open-h3-ir in its own venv so its dependencies never move ComfyUI's pins; the worker runs `h3ir serve` as a child
RUN python3.12 -m venv /opt/h3ir \
    && /opt/h3ir/bin/pip install "pip==25.3" "setuptools>=77" "packaging>=24.2" wheel \
    && /opt/h3ir/bin/pip install "git+https://github.com/ruashots/open-h3-ir.git@${H3IR_COMMIT}" \
    && /opt/h3ir/bin/h3ir --help > /dev/null

COPY comfy/extra_model_paths.yaml ComfyUI/extra_model_paths.yaml
COPY workflows/ /app/workflows/
COPY src/ /app/src/
# official prompt-writing skill, fetched at a pinned commit and hash-checked (not vendored: upstream has no license file)
COPY scripts/fetch_skill.sh /app/scripts/
RUN rm -rf /app/src/skill && sh /app/scripts/fetch_skill.sh /app/src/skill

RUN pip freeze > /app/requirements.lock && /opt/h3ir/bin/pip freeze > /app/requirements-h3ir.lock \
    && python - <<'PY'
import torch, importlib.metadata as m
print("torch", torch.__version__, "cuda", torch.version.cuda, "| comfy-kitchen", m.version("comfy-kitchen"), "| runpod", m.version("runpod"),
      "| boto3", m.version("boto3"))
assert torch.version.cuda.startswith("13."), torch.version.cuda
import sys; sys.path.insert(0, "/app/src"); import handler  # imports and the workflow template must load
PY

# last, so a new tag only rebuilds this layer: image tag echoed in output.build; H3_COMFY_ARGS = ComfyUI flags (template env)
ARG BUILD_REF=dev
ENV H3_BUILD=${BUILD_REF}

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "/app/src/handler.py"]

# tiny stage used by CI to pull the lock files out of the image
FROM scratch AS artifacts
COPY --from=runtime /app/requirements.lock /requirements.lock
COPY --from=runtime /app/requirements-h3ir.lock /requirements-h3ir.lock
