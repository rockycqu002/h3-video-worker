"""Canvas maths, keyframe preprocessing and the ComfyUI graph for MiniMax H3 (DaSiWa Hybrid 8-step, Turbo baked in).

Ported from the AutoDL web app (h3_deploy/app/workflow.py); only the DaSiWa path is kept.
"""
import copy, json, os

from PIL import Image, ImageOps

WORKFLOW_PATH = os.environ.get("H3_WORKFLOW", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflows", "fl2va_dasiwa_api.json"))
ASPECTS = {"21:9": 21 / 9, "16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1.0, "3:4": 3 / 4, "9:16": 9 / 16}
QUALITIES = {"768": 768, "640": 640, "544": 544, "416": 416}   # short-edge pixels; 768 = native 768p
MAX_LONG_EDGE = 1344
FPS = 24

_TEMPLATE = None


def template():
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = json.load(open(WORKFLOW_PATH))
    return _TEMPLATE


def frames_for_seconds(sec: float) -> int:
    """H3 generates on a 17k+5 frame grid at 24 fps; snap up like the official template."""
    n = max(5, round(sec * FPS))
    return n + (5 - n % 17) % 17


def resolution(aspect: str, quality: str, ratio: float | None = None):
    """Short edge = quality preset, long edge follows the aspect ratio, capped at H3's native 1344.
    `ratio` (w/h) overrides the named aspect — used for "auto" (follow the keyframe)."""
    ar = ratio if ratio else ASPECTS[aspect]
    short = QUALITIES[quality]
    long = short * max(ar, 1 / ar)
    if long > MAX_LONG_EDGE:  # keep the aspect ratio, shrink both edges
        short, long = short * MAX_LONG_EDGE / long, MAX_LONG_EDGE
    w, h = (long, short) if ar >= 1 else (short, long)
    return int(round(w / 32) * 32), int(round(h / 32) * 32)


def normalize(im: Image.Image) -> Image.Image:
    """EXIF orientation applied, alpha composited onto white, RGB."""
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[-1]); return bg
    return im.convert("RGB")


def cover_crop(im: Image.Image, w: int, h: int) -> Image.Image:
    """Center-crop to the canvas aspect and resize (never stretch) — ComfyUI would stretch the first frame otherwise."""
    iw, ih = im.size
    target = w / h
    if abs(iw / ih - target) > 1e-3:
        if iw / ih > target:  # too wide
            nw = int(round(ih * target)); im = im.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
        else:                 # too tall
            nh = int(round(iw / target)); im = im.crop((0, (ih - nh) // 2, iw, (ih - nh) // 2 + nh))
    return im.resize((w, h), Image.LANCZOS)


def build(prompt: str, width: int, height: int, seconds: float, seed: int, steps: int, prefix: str,
          first_frame: str | None = None, last_frame: str | None = None) -> dict:
    """first_frame / last_frame are file names inside ComfyUI's input directory."""
    wf = copy.deepcopy(template())
    wf["cond"]["inputs"].update(prompt=prompt, width=width, height=height, length=frames_for_seconds(seconds))
    wf["noise"]["inputs"]["noise_seed"] = seed
    wf["sigmas"]["inputs"]["steps"] = steps
    wf["save"]["inputs"]["filename_prefix"] = prefix
    if first_frame:
        wf["img_first"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        wf["cond"]["inputs"]["first_frame"] = ["img_first", 0]
    if last_frame:
        wf["img_last"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
        wf["cond"]["inputs"]["last_frame"] = ["img_last", 0]
    return wf
