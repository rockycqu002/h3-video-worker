# h3-video-worker

RunPod Serverless worker: **MiniMax H3** text / keyframe → video with stereo audio (768p, 24 fps, 4–15 s), using the
community **DaSiWa Hybrid 8-step** checkpoint (Turbo baked in) on **RTX 4090 / CUDA 13**, served through headless ComfyUI.
Prompt expansion (open-h3-ir, with the official `h3-prompt-writing` skill as fallback) runs inside the worker; videos go
to Cloudflare R2. The caller is a Cloudflare Worker. Design and decisions: [`docs/PLAN.md`](docs/PLAN.md). Caller guide: [`docs/API_CALLER.md`](docs/API_CALLER.md). Test report: [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md).

Endpoints: production `zobq85s4yao4r6` (`h3-video-4090`, template `heeq5jw6rw`, 0–3 workers); staging `ekb0yn2gdsmyjo` (`h3-video-4090-test`, template `tjw3t06amu`, 1 worker). Release flow: tag → CI → `template-update` staging → drain (max 0 → 1) → smoke → `template-update` production → drain.

```json
POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run
{"input": {"request": "a girl turns around and smiles on a rainy neon street", "seconds": 5,
           "first_frame_url": "https://<r2 presigned>"}, "webhook": "https://<cf-worker>/runpod-hook"}
→ output: {"video_key": "videos/<job_id>.mp4", "width": 768, "height": 1344, "frames": 124, "seed": 123,
           "prompt": "...", "rewrite": {"engine": "openh3ir", ...}, "timing": {...}}
→ or FAILED with error "<code>: <message>", code ∈ bad_input | rewrite_failed | comfy_rejected | oom |
  generate_timeout | upload_failed | internal
```

## Layout

| path | purpose |
|---|---|
| `Dockerfile` | CUDA 13.0.1 base (digest) → torch 2.14.0+cu130 → ComfyUI 0.37.0 @ `88ab4a0` → open-h3-ir 0.4.1 in `/opt/h3ir`; **no weights** |
| `constraints.txt` | pip pins, shared with qwen-image-edit-4090 (validated on RunPod 4090s) |
| `comfy/extra_model_paths.yaml` | ComfyUI reads weights from `/runpod-volume/h3/models` |
| `workflows/fl2va_dasiwa_api.json` | API-format graph: UNET → SigmaShift(9, 4) → euler/simple 8 steps → video + audio VAE → mp4 |
| `src/handler.py` | boot (ComfyUI + h3ir + page-cache prefetch), input validation, rewrite ‖ cold start, generation, R2 upload, error codes |
| `src/workflow.py` | frame grid, canvas size, cover-crop, graph builder |
| `src/rewrite.py` | open-h3-ir client + skill fallback (OpenRouter) |
| `src/storage.py` | allow-listed keyframe download, R2 upload |
| `scripts/fetch_skill.sh` | fetches the official `h3-prompt-writing` skill (MiniMax-AI/MiniMax-H3 @ pinned commit, SHA-256 checked) into `src/skill/` — not vendored |
| `scripts/fetch_models.py` | fills the network volume at pinned HF revisions, SHA-256 verified |
| `.github/workflows/build.yml` | `v*` tag → build and push `ghcr.io/<owner>/h3-video-worker:<tag>` |
| `tests/` | CPU unit tests |

## Endpoint environment

| var | meaning |
|---|---|
| `OPENROUTER_API_KEY` | rewrite (secret) |
| `OPENROUTER_MODEL` | default `qwen/qwen3-vl-235b-a22b-instruct` |
| `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | output bucket; token scoped to that bucket (secrets) |
| `R2_PREFIX` | default `videos` |
| `FRAME_URL_ALLOW` | comma-separated keyframe hosts; `.example.com` also matches subdomains |
| `H3_COMFY_ARGS` | ComfyUI flags, default `--reserve-vram 1 --disable-nvml-pressure --fast-disk` (`--fast-disk` is required on RunPod 4090s: 43.9 GB RAM limit) |
| `H3_JOB_DEADLINE_S` | default 1740 (endpoint executionTimeout 1800 s) |
| `H3_PREFETCH` | `1` (default) reads the weights into the page cache at boot |

## Local checks

```
python -m venv .venv && .venv/bin/pip install pillow pytest requests
sh scripts/fetch_skill.sh
.venv/bin/python -m pytest tests -q
```
