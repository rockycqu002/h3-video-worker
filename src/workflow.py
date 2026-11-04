"""Canvas maths, keyframe preprocessing and the ComfyUI graph for MiniMax H3 (DaSiWa Hybrid 8-step, Turbo baked in).

Ported from the AutoDL web app (h3_deploy/app/workflow.py); only the DaSiWa path is kept. The same DaSiWa Hybrid
checkpoint serves T2V / keyframes (MiniMaxH3ImageToVideo) and Ref2VA (MiniMaxH3ReferenceToVideo, see build()).
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


REF_IMAGE_MAX_EDGE = 2048   # the node scales references to the generation's pixel area ("match"), so more is wasted


def fit_reference(im: Image.Image) -> Image.Image:
    """Caps a reference image's long edge (aspect kept, never upscaled, never cropped)."""
    if max(im.size) <= REF_IMAGE_MAX_EDGE:
        return im
    im = im.copy()
    im.thumbnail((REF_IMAGE_MAX_EDGE, REF_IMAGE_MAX_EDGE), Image.LANCZOS)
    return im


def build(prompt: str, width: int, height: int, seconds: float, seed: int, steps: int, prefix: str,
          first_frame: str | None = None, last_frame: str | None = None,
          ref_images: list[str] | None = None, ref_videos: list[str] | None = None, ref_video_audio: bool = False) -> dict:
    """first_frame / last_frame / ref_images / ref_videos are file names inside ComfyUI's input directory.

    With ref_images or ref_videos the conditioning node becomes MiniMaxH3ReferenceToVideo (Ref2VA): the prompt calls
    them <Picture 1..n> and <Video 1..k> in list order; with ref_video_audio each video's soundtrack is attached too
    and becomes <Audio k>. Keyframes and references do not mix (the handler rejects that)."""
    wf = copy.deepcopy(template())
    wf["cond"]["inputs"].update(prompt=prompt, width=width, height=height, length=frames_for_seconds(seconds))
    wf["noise"]["inputs"]["noise_seed"] = seed
    wf["sigmas"]["inputs"]["steps"] = steps
    wf["save"]["inputs"]["filename_prefix"] = prefix
    if ref_images or ref_videos:
        if first_frame or last_frame:
            raise ValueError("keyframes and references cannot be combined")
        _reference_cond(wf, ref_images or [], ref_videos or [], ref_video_audio)
        return wf
    if first_frame:
        wf["img_first"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        wf["cond"]["inputs"]["first_frame"] = ["img_first", 0]
    if last_frame:
        wf["img_last"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
        wf["cond"]["inputs"]["last_frame"] = ["img_last", 0]
    return wf


def _reference_cond(wf: dict, images: list[str], videos: list[str], video_audio: bool):
    """Replaces node "cond" in place; outputs keep their order (0 = positive, 1 = AV latent), so the guider and the
    sampler stay wired. Autogrow inputs are addressed as "<group>.<prefix><i>" with i counted from 0."""
    old = wf["cond"]["inputs"]
    inputs = {"clip": old["clip"], "vae": old["vae"], "audio_vae": ["vae_a", 0], "prompt": old["prompt"],
              "width": old["width"], "height": old["height"], "length": old["length"], "ref_image_size": "match"}
    for i, name in enumerate(images):
        wf[f"ref_img_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        inputs[f"ref_images.ref_image_{i}"] = [f"ref_img_{i}", 0]
    for k, name in enumerate(videos):
        wf[f"ref_vid_{k}"] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        wf[f"ref_vid_{k}_parts"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"ref_vid_{k}", 0]}}
        inputs[f"ref_videos.ref_video_{k}"] = [f"ref_vid_{k}_parts", 0]
        if video_audio:
            inputs[f"ref_video_audios.ref_video_audio_{k}"] = [f"ref_vid_{k}_parts", 1]
    wf["cond"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": inputs}
