#!/usr/bin/env python3
"""Populate the RunPod Network Volume with the H3 weights at pinned Hugging Face revisions and verify SHA-256.

Run once on a temporary Pod that has the volume mounted (Pods mount it at /workspace, serverless workers at /runpod-volume):
  pip install huggingface_hub hf_xet && python fetch_models.py --dest /workspace/h3/models
Re-running is safe: files whose size and hash already match are skipped. Writes <dest>/MANIFEST.json at the end.
Revisions and hashes were read from the HF API on 2026-09-29; the DaSiWa file is the one validated on AutoDL.
"""
import argparse, hashlib, json, os, shutil, sys, time

from huggingface_hub import hf_hub_download

COMFY_ORG = ("Comfy-Org/MiniMax-H3", "e5eb578a89295337b8ff433a035929ce0279e0b6")
DASIWA = ("brurpo/DaSiWa-MiniMax-H3-Hybrid", "f831fe1439fae8fcdacfb4047a20753541727ba0")
# local path under dest -> (repo, revision, path in repo, bytes, sha256)
FILES = {
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors":
        (*COMFY_ORG, "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15687142551, "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"),
    "diffusion_models/DasiwaMinimaxH3_dasiwaHybrid8turboV1.safetensors":
        (*DASIWA, "DasiwaMinimaxH3_dasiwaHybrid8turboV1.safetensors", 20967642441, "e0441d26414f6e0c28f43d580e6cc56fad424da0fa4d261b698ca73188aa6332"),
    "vae/minimax_h3_video_vae_fp16.safetensors":
        (*COMFY_ORG, "vae/minimax_h3_video_vae_fp16.safetensors", 5207808496, "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"),
    "vae/minimax_h3_audio_vae_fp32.safetensors":
        (*COMFY_ORG, "vae/minimax_h3_audio_vae_fp32.safetensors", 605254808, "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest, rel):
    repo, rev, src, size, expected = FILES[rel]
    final = os.path.join(dest, rel)
    if os.path.isfile(final) and os.path.getsize(final) == size and sha256(final) == expected:
        print(f"{rel}: already present, hash ok", flush=True)
        return {"path": rel, "repo": repo, "revision": rev, "bytes": size, "sha256": expected}
    t0 = time.time()
    staging = os.path.join(dest, ".staging")
    path = hf_hub_download(repo, src, revision=rev, local_dir=staging)
    got = sha256(path)
    print(f"{rel}: {os.path.getsize(path) / 1e9:.2f} GB in {time.time() - t0:.0f}s sha256={got}", flush=True)
    if got != expected or os.path.getsize(path) != size:
        sys.exit(f"SHA-256/size MISMATCH for {rel}: expected {expected} ({size} bytes)")
    os.makedirs(os.path.dirname(final), exist_ok=True)
    shutil.move(path, final)
    return {"path": rel, "repo": repo, "revision": rev, "bytes": size, "sha256": got}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="e.g. /workspace/h3/models on a Pod")
    ap.add_argument("--only", action="append", help="local relative path(s); default all")
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    recs = [fetch(a.dest, rel) for rel in (a.only or list(FILES))]
    shutil.rmtree(os.path.join(a.dest, ".staging"), ignore_errors=True)
    if not a.only:
        with open(os.path.join(a.dest, "MANIFEST.json"), "w") as f:
            json.dump({"written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": recs}, f, indent=1)
        print("manifest written", flush=True)


if __name__ == "__main__":
    main()
