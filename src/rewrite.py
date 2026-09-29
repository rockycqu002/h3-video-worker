"""Prompt expansion: plain user request -> H3-format prompt.

Primary engine: open-h3-ir (`h3ir serve` on 127.0.0.1:8420, started by handler.py) for T2V / I2V / L2V.
Fallback engine "skill": one OpenRouter chat call with the official h3-prompt-writing skill as system prompt.
It is used for FL2V (open-h3-ir 0.4.1 crashes on two frame anchors) and whenever open-h3-ir fails.
"""
import base64, io, os, re, tempfile, time

import requests
from PIL import Image

H3IR_URL = os.environ.get("H3IR_URL", "http://127.0.0.1:8420")
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-vl-235b-a22b-instruct")
CREATIVITY = ("restrained", "balanced", "bold", "extreme")
SKILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skill")
REQUIRED_FIELDS = ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:")
MODE_NAMES = {"t2v": "T2VA", "i2v": "I2VA", "l2v": "L2VA", "fl2v": "FL2VA"}


class RewriteError(Exception):
    pass


def mode_of(first, last):
    return "fl2v" if (first is not None and last is not None) else "i2v" if first is not None else "l2v" if last is not None else "t2v"


def h3ir_ready(timeout=3):
    try:
        r = requests.get(f"{H3IR_URL}/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("ok", False)
    except Exception:
        return False


# ----------------------------------------------------------------------------- open-h3-ir
def openh3ir(request, seconds, aspect, first, last, creativity, tmp_dir, deadline_s=240):
    """first / last: PIL images (already cover-cropped to the canvas) or None."""
    assets, paths = [], []
    try:
        for role, im in (("frame_anchor_first", first), ("frame_anchor_last", last)):
            if im is not None:
                fd, p = tempfile.mkstemp(prefix="h3ir_", suffix=".png", dir=tmp_dir); os.close(fd)
                im.save(p, format="PNG", compress_level=1)
                assets.append({"path": p, "role": role}); paths.append(p)
        body = {"intent": request.strip(), "seconds": seconds, "creativity": creativity}
        if seconds < 6:
            body["shots"] = 1   # the writer model tends to cut inside the 1.2 s floor on short 2-shot clips
        if assets:
            body["assets"] = assets
        elif aspect in ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"):
            body["aspect"] = aspect
        r = requests.post(f"{H3IR_URL}/v1/briefs", json=body, timeout=deadline_s)
        try:
            j = r.json()
        except ValueError:
            raise RewriteError(f"open-h3-ir HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code not in (200, 201):
            d = j.get("detail", j) if isinstance(j, dict) else j
            msg = d.get("message", str(d)) if isinstance(d, dict) else str(d)
            raise RewriteError(f"open-h3-ir HTTP {r.status_code}: {msg[:300]}")
    except requests.RequestException as e:
        raise RewriteError(f"open-h3-ir unreachable: {type(e).__name__}")
    finally:
        for p in paths:
            try: os.remove(p)
            except OSError: pass
    ir = j["ir"]
    warns = [f"{d.get('rule')}: {d.get('message')}" for d in ir.get("diagnostics", []) if d.get("severity") in ("WARN", "ERROR")]
    return {"prompt": ir["prompt"], "status": j.get("status"), "fallback_reason": j.get("fallback_reason"),
            "tokens": ir.get("prompt_tokens"), "warnings": warns[:6]}


# ----------------------------------------------------------------------------- skill fallback
def _system_prompt():
    skill = open(os.path.join(SKILL_DIR, "SKILL.md")).read()
    base = open(os.path.join(SKILL_DIR, "base-en.txt")).read()
    return (f"{skill}\n\n---\n# references/base-en.txt\n\n{base}\n\n---\n"
            "You are the prompt rewriter of a video generation service. Apply the skill above to the user's request and "
            "return ONLY the final H3 prompt: the alignment instruction line (if the mode has one), a blank line, then the "
            "three core fields in order. No preamble, no markdown fences, no commentary.")


def _data_url(im, edge=1024):
    im = im.copy(); im.thumbnail((edge, edge))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def clean_output(text):
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*\n", "", t); t = re.sub(r"\n```\s*$", "", t)
    return t.strip()


def validate(prompt, mode):
    missing = [f for f in REQUIRED_FIELDS if f not in prompt]
    if missing:
        raise RewriteError(f"skill output misses {', '.join(missing)}")
    if mode != "t2v" and "Picture 1" not in prompt.split("integrated_multimodal_description:")[0]:
        raise RewriteError("skill output has no keyframe alignment line")


def skill(request, mode, seconds, frames, fps, first, last, api_key, attempts=3, deadline_s=240):
    if not api_key:
        raise RewriteError("OPENROUTER_API_KEY is not set")
    eff = f"{frames / fps:.2f}"
    head = (f"Mode: {MODE_NAMES[mode]}. Requested duration: {seconds:g} seconds; effective video duration S.SS = {eff} "
            f"({frames} frames at {fps} fps) — use {eff} wherever the guide asks for S.SS and keep every timestamp within it.\n")
    if mode == "fl2v":
        head += "The first attached image is Picture 1 (first frame, 0.00 s); the second is Picture 2 (last frame).\n"
    elif mode in ("i2v", "l2v"):
        head += "The attached image is Picture 1 (" + ("first frame, 0.00 s" if mode == "i2v" else "last frame") + ").\n"
    content = [{"type": "text", "text": head + "\nUser request:\n" + request.strip()}]
    for im in (first, last):
        if im is not None:
            content.append({"type": "image_url", "image_url": {"url": _data_url(im)}})
    body = {"model": OPENROUTER_MODEL, "temperature": 0.6, "max_tokens": 3000,
            "messages": [{"role": "system", "content": _system_prompt()}, {"role": "user", "content": content}]}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "X-Title": "h3-video-worker"}
    t_end, last_err = time.time() + deadline_s, None
    for i in range(attempts):
        if time.time() > t_end:
            break
        try:
            r = requests.post(OPENROUTER_URL, json=body, headers=headers, timeout=max(10, t_end - time.time()))
            if r.status_code != 200:
                raise RewriteError(f"OpenRouter HTTP {r.status_code}: {r.text[:200]}")
            j = r.json()
            prompt = clean_output(j["choices"][0]["message"]["content"] or "")
            validate(prompt, mode)
            usage = j.get("usage") or {}
            return {"prompt": prompt, "status": "ok", "fallback_reason": None,
                    "tokens": usage.get("total_tokens"), "warnings": []}
        except (requests.RequestException, KeyError, IndexError, ValueError, RewriteError) as e:
            last_err = e if isinstance(e, RewriteError) else RewriteError(f"{type(e).__name__}: {str(e)[:200]}")
            time.sleep(min(2 ** i, 5))
    raise last_err or RewriteError("skill rewrite timed out")


# ----------------------------------------------------------------------------- entry point
def rewrite(request, seconds, aspect, frames, fps, first, last, creativity, api_key, tmp_dir, h3ir_wait_s=90):
    """Returns {"prompt", "engine", "status", "fallback_reason", "warnings", "tokens", "seconds"}; raises RewriteError."""
    t0 = time.time()
    mode = mode_of(first, last)
    reason = None
    if mode == "fl2v":
        reason = "open-h3-ir 0.4.1 cannot compile first+last-frame briefs"
    else:
        t_wait = time.time() + h3ir_wait_s          # h3ir starts together with the worker; give it a moment on cold boot
        while not h3ir_ready() and time.time() < t_wait:
            time.sleep(1)
        try:
            r = openh3ir(request, seconds, aspect, first, last, creativity, tmp_dir)
            return {**r, "engine": "openh3ir", "seconds": round(time.time() - t0, 1)}
        except (RewriteError, KeyError, TypeError) as e:
            reason = f"open-h3-ir failed: {str(e)[:200]}"
    r = skill(request, mode, seconds, frames, fps, first, last, api_key)
    return {**r, "engine": "skill", "fallback_reason": reason, "seconds": round(time.time() - t0, 1)}
