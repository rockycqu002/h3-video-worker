"""CPU-only unit tests: canvas maths, input validation, graph construction, rewrite routing and the handler's
orchestration / error codes with ComfyUI, R2 and OpenRouter stubbed out.

Run:  python -m pytest tests -q
"""
import os, sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import handler as h  # noqa: E402
import rewrite, storage, workflow  # noqa: E402


# ----------------------------------------------------------------------------- workflow
def test_frames_grid():
    assert workflow.frames_for_seconds(5) == 124
    assert workflow.frames_for_seconds(4) == 107
    assert workflow.frames_for_seconds(15) == 362
    for s in (4, 5, 6.5, 10, 15):
        assert (workflow.frames_for_seconds(s) - 5) % 17 == 0


def test_resolution_matches_autodl():
    assert workflow.resolution("16:9", "768") == (1344, 768)
    assert workflow.resolution("9:16", "768") == (768, 1344)
    assert workflow.resolution("1:1", "768") == (768, 768)
    w, h_ = workflow.resolution("16:9", "416")
    assert w % 32 == 0 and h_ % 32 == 0 and w > h_


def test_cover_crop_never_stretches():
    im = Image.new("RGB", (1000, 1000), (255, 0, 0))
    for x in range(250):                      # a blue band on the left that a centre crop to 16:9 keeps
        for y in range(1000):
            im.putpixel((x, y), (0, 0, 255))
    out = workflow.cover_crop(im, 1344, 768)
    assert out.size == (1344, 768)
    assert out.getpixel((0, 384))[2] > 200    # cropped top/bottom, not the sides


def test_normalize_alpha_and_exif():
    rgba = Image.new("RGBA", (8, 8), (0, 0, 255, 0))
    assert workflow.normalize(rgba).getpixel((0, 0)) == (255, 255, 255)


def test_graph_dasiwa_only():
    g = workflow.build("P", 1344, 768, 5, 42, 8, "job/abc")
    assert "lora" not in g and g["unet"]["inputs"]["unet_name"].startswith("Dasiwa")
    assert g["guider"]["inputs"]["model"] == ["shift", 0] and g["sigmas"]["inputs"]["model"] == ["shift", 0]
    assert g["sampler"]["inputs"]["sampler_name"] == "euler"
    assert (g["shift"]["inputs"]["shift_video"], g["shift"]["inputs"]["shift_audio"]) == (9.0, 4.0)
    assert g["cond"]["inputs"]["length"] == 124 and g["noise"]["inputs"]["noise_seed"] == 42
    assert "first_frame" not in g["cond"]["inputs"]
    g = workflow.build("P", 1344, 768, 5, 1, 8, "job/abc", "a.png", "b.png")
    assert g["cond"]["inputs"]["first_frame"] == ["img_first", 0] and g["img_last"]["inputs"]["image"] == "b.png"


def test_template_not_mutated():
    workflow.build("X", 64, 64, 4, 1, 6, "p", "a.png")
    assert "img_first" not in workflow.template() and workflow.template()["cond"]["inputs"]["prompt"] == "PROMPT"


# ----------------------------------------------------------------------------- input
def test_parse_defaults():
    p = h.parse_input({"request": "  a cat  "})
    assert p["request"] == "a cat" and p["seconds"] == 5.0 and p["aspect"] == "auto" and p["quality"] == "768"
    assert p["steps"] == 8 and p["creativity"] == "balanced" and p["skip_rewrite"] is False
    assert 0 <= p["seed"] < 2**32


@pytest.mark.parametrize("bad", [
    None, {}, {"request": ""}, {"request": "x" * 7000}, {"request": "x", "seconds": 3}, {"request": "x", "seconds": 16},
    {"request": "x", "seconds": "abc"}, {"request": "x", "steps": 20}, {"request": "x", "aspect": "2:1"},
    {"request": "x", "quality": "1080"}, {"request": "x", "creativity": "wild"}, {"request": "x", "skip_rewrite": "yes"},
    {"request": "x", "seed": -1}, {"request": "x", "seed": True},
])
def test_parse_rejects(bad):
    with pytest.raises(storage.BadInput):
        h.parse_input(bad)


def test_canvas_auto_follows_keyframe():
    p = h.parse_input({"request": "x"})
    assert h.canvas(p, None, None) == (1344, 768)
    portrait = Image.new("RGB", (900, 1600))
    assert h.canvas(p, portrait, None) == (768, 1344)
    assert h.canvas(p, None, portrait) == (768, 1344)       # last frame alone also drives "auto"
    p = h.parse_input({"request": "x", "aspect": "1:1"})
    assert h.canvas(p, portrait, None) == (768, 768)


# ----------------------------------------------------------------------------- storage
def test_host_allowlist(monkeypatch):
    monkeypatch.setenv("FRAME_URL_ALLOW", "frames.example.com, .r2.cloudflarestorage.com")
    allow = storage.allowed_hosts()
    assert storage.host_allowed("frames.example.com", allow)
    assert storage.host_allowed("acct.r2.cloudflarestorage.com", allow)
    assert not storage.host_allowed("evil.com", allow)
    assert not storage.host_allowed("frames.example.com.evil.com", allow)
    assert not storage.host_allowed("r2.cloudflarestorage.com.evil", allow)


@pytest.mark.parametrize("url", ["http://frames.example.com/a.png", "https://169.254.169.254/latest", "ftp://x/y", 123])
def test_fetch_rejects_before_network(monkeypatch, url):
    monkeypatch.setenv("FRAME_URL_ALLOW", "frames.example.com")
    monkeypatch.setattr(storage.requests, "get", lambda *a, **k: pytest.fail("must not fetch"))
    with pytest.raises(storage.BadInput):
        storage.fetch_image("first_frame_url", url)


def test_decode_image_rejects_garbage():
    for blob in (b"", b"GIF89a" + b"\0" * 50, b"\xff\xd8\xff" + b"\0" * 50):
        with pytest.raises(storage.BadInput):
            storage.decode_image("f", blob)


def test_video_key():
    assert storage.video_key("abc-123_x") == "videos/abc-123_x.mp4"
    for bad in ("", "../etc/passwd", "a/b", "a b"):
        with pytest.raises(ValueError):
            storage.video_key(bad)


# ----------------------------------------------------------------------------- rewrite
class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p, self.text = status, payload, str(payload)

    def json(self):
        return self._p


GOOD = ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark "
        "of the target video; Picture 2 (from Shot 1) aligns with the 5.17-second mark of the target video.\n\n"
        "integrated_multimodal_description: [Shot 1] ...\n\noverall_soundscape: rain\n\nnon_diegetic_music: none")


def test_fl2v_goes_to_skill_without_h3ir(monkeypatch):
    calls = []
    monkeypatch.setattr(rewrite, "h3ir_ready", lambda *a, **k: pytest.fail("h3ir must not be consulted for FL2V"))

    def post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return FakeResp(200, {"choices": [{"message": {"content": "```text\n" + GOOD + "\n```"}}], "usage": {"total_tokens": 900}})
    monkeypatch.setattr(rewrite.requests, "post", post)
    im = Image.new("RGB", (64, 64))
    r = rewrite.rewrite("walk to the door", 5, "auto", 124, 24, im, im, "balanced", "k", "/tmp")
    assert r["engine"] == "skill" and r["prompt"] == GOOD and "first+last" in r["fallback_reason"]
    user = calls[0]["messages"][1]["content"]
    assert "FL2VA" in user[0]["text"] and "5.17" in user[0]["text"] and sum(c["type"] == "image_url" for c in user) == 2
    assert "integrated_multimodal_description" in calls[0]["messages"][0]["content"]   # skill + base guide in the system prompt


def test_skill_rejects_malformed_then_fails(monkeypatch):
    monkeypatch.setattr(rewrite.time, "sleep", lambda s: None)
    monkeypatch.setattr(rewrite.requests, "post", lambda *a, **k: FakeResp(200, {"choices": [{"message": {"content": "just prose"}}]}))
    with pytest.raises(rewrite.RewriteError, match="misses"):
        rewrite.skill("x", "t2v", 5, 124, 24, None, None, "k")


def test_skill_needs_key():
    with pytest.raises(rewrite.RewriteError, match="OPENROUTER_API_KEY"):
        rewrite.skill("x", "t2v", 5, 124, 24, None, None, "")


def test_h3ir_failure_falls_back_to_skill(monkeypatch):
    monkeypatch.setattr(rewrite, "h3ir_ready", lambda *a, **k: True)
    t2v = GOOD.split("\n\n", 1)[1]

    def post(url, json=None, headers=None, timeout=None):
        if url.endswith("/v1/briefs"):
            return FakeResp(500, {"detail": {"message": "boom"}})
        return FakeResp(200, {"choices": [{"message": {"content": t2v}}]})
    monkeypatch.setattr(rewrite.requests, "post", post)
    r = rewrite.rewrite("a cat", 5, "16:9", 124, 24, None, None, "balanced", "k", "/tmp")
    assert r["engine"] == "skill" and "open-h3-ir failed" in r["fallback_reason"] and r["prompt"] == t2v


def test_h3ir_success(monkeypatch, tmp_path):
    monkeypatch.setattr(rewrite, "h3ir_ready", lambda *a, **k: True)
    seen = {}

    def post(url, json=None, headers=None, timeout=None):
        seen.update(json)
        assert all(os.path.exists(a["path"]) for a in json.get("assets", []))
        return FakeResp(201, {"status": "ok", "ir": {"prompt": "IR PROMPT", "prompt_tokens": 800, "diagnostics": [{"severity": "WARN", "rule": "T1", "message": "m"}]}})
    monkeypatch.setattr(rewrite.requests, "post", post)
    r = rewrite.rewrite("a cat", 5, "auto", 124, 24, Image.new("RGB", (32, 32)), None, "bold", "k", str(tmp_path))
    assert r["engine"] == "openh3ir" and r["prompt"] == "IR PROMPT" and r["warnings"] == ["T1: m"]
    assert seen["shots"] == 1 and seen["assets"][0]["role"] == "frame_anchor_first" and "aspect" not in seen
    assert not list(tmp_path.iterdir())          # temp keyframes removed


# ----------------------------------------------------------------------------- handler orchestration
@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    monkeypatch.setattr(h, "IN_DIR", str(tmp_path)); monkeypatch.setattr(h, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(h, "_state", {"boot_t0": 0, "comfy_ready_s": 1, "jobs": 0, "models_missing": [], "prefetch_s": None})
    monkeypatch.setattr(h, "wait_comfy", lambda deadline: None)
    monkeypatch.setattr(h, "comfy_alive", lambda: True)
    monkeypatch.setattr(h, "vram_used_mib", lambda: 20000)
    monkeypatch.setattr(h, "_progress", lambda job, msg: None)
    monkeypatch.setattr(storage, "fetch_image", lambda field, url: Image.new("RGB", (900, 1600), (9, 9, 9)))
    state = {"graph": None, "uploaded": None}

    def run_graph(job, graph, deadline):
        state["graph"] = graph
        p = tmp_path / "out.mp4"; p.write_bytes(b"mp4"); return str(p), 190.0
    monkeypatch.setattr(h, "run_graph", run_graph)

    def upload(path, key):
        state["uploaded"] = key; return 3
    monkeypatch.setattr(storage, "upload_video", upload)
    monkeypatch.setattr(storage, "presign_video", lambda key, seconds: f"https://r2.example/{key}?sig=1")
    monkeypatch.setattr(rewrite, "rewrite", lambda *a, **k: {"prompt": "REWRITTEN", "engine": "openh3ir", "status": "ok",
                                                              "fallback_reason": None, "warnings": [], "tokens": 1, "seconds": 2.0})
    return state


def test_handler_happy_path(stubbed):
    out = h.handler({"id": "job-1", "input": {"request": "a cat", "first_frame_url": "https://x/a.png", "seed": 7}})
    assert "error" not in out, out
    assert out["video_key"] == "videos/job-1.mp4" == stubbed["uploaded"]
    assert (out["width"], out["height"]) == (768, 1344) and out["mode"] == "i2v" and out["seed"] == 7
    assert out["prompt"] == "REWRITTEN" and out["rewrite"]["engine"] == "openh3ir" and "prompt" not in out["rewrite"]
    assert stubbed["graph"]["cond"]["inputs"]["prompt"] == "REWRITTEN"
    assert stubbed["graph"]["img_first"]["inputs"]["image"] == "job-1_first.png"
    assert out["timing"]["generate"] == 190.0
    assert out["video_url"] == "https://r2.example/videos/job-1.mp4?sig=1" and "refs" not in out


def test_handler_skip_rewrite(stubbed, monkeypatch):
    monkeypatch.setattr(rewrite, "rewrite", lambda *a, **k: pytest.fail("must not rewrite"))
    out = h.handler({"id": "j2", "input": {"request": "FINAL PROMPT", "skip_rewrite": True}})
    assert out["prompt"] == "FINAL PROMPT" and out["rewrite"]["engine"] == "none" and out["mode"] == "t2v"


def test_handler_error_codes(stubbed, monkeypatch):
    assert h.handler({"id": "j3", "input": {"request": ""}})["error"].startswith("bad_input:")
    assert h.handler({"id": "../x", "input": {"request": "a"}})["error"].startswith("internal:")

    def boom(*a, **k):
        raise rewrite.RewriteError("OpenRouter HTTP 402")
    monkeypatch.setattr(rewrite, "rewrite", boom)
    assert h.handler({"id": "j4", "input": {"request": "a"}})["error"] == "rewrite_failed: OpenRouter HTTP 402"


def test_handler_oom_refreshes_worker(stubbed, monkeypatch):
    def oom(job, graph, deadline):
        raise h.execution_error({"messages": [["execution_error", {"node_type": "SamplerCustomAdvanced", "exception_type": "torch.OutOfMemoryError",
                                                                   "exception_message": "CUDA out of memory"}]]})
    monkeypatch.setattr(h, "run_graph", oom)
    out = h.handler({"id": "j5", "input": {"request": "a", "skip_rewrite": True}})
    assert out["error"].startswith("oom:") and out["refresh_worker"] is True


def test_handler_upload_failure(stubbed, monkeypatch):
    def fail(path, key):
        raise RuntimeError("403 AccessDenied")
    monkeypatch.setattr(storage, "upload_video", fail)
    out = h.handler({"id": "j6", "input": {"request": "a", "skip_rewrite": True}})
    assert out["error"].startswith("upload_failed:") and "refresh_worker" not in out


def test_handler_models_missing(stubbed):
    h._state["models_missing"] = ["vae/x.safetensors"]
    assert "model files missing" in h.handler({"id": "j7", "input": {"request": "a"}})["error"]


# ----------------------------------------------------------------------------- Ref2VA (v0.2.0)
def make_video(path, seconds=2.5, fps=24, size=(96, 160), audio=False):
    """Tiny real mp4 through PyAV (no numpy needed): solid frames, optional silent AAC track."""
    import fractions
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as c:
        v = c.add_stream("libx264" if "libx264" in av.codecs_available else "mpeg4", rate=fps)
        v.width, v.height, v.pix_fmt = size[0], size[1], "yuv420p"
        a = c.add_stream("aac", rate=32000) if audio else None
        if a is not None:
            a.layout = "mono"
        for i in range(int(round(seconds * fps))):
            for pkt in v.encode(av.VideoFrame.from_image(Image.new("RGB", size, (i * 9 % 255, 40, 90)))):
                c.mux(pkt)
        for pkt in v.encode():
            c.mux(pkt)
        if a is not None:
            for start in range(0, int(seconds * 32000), 1024):
                fr = av.AudioFrame(format="fltp", layout="mono", samples=1024)
                for plane in fr.planes:
                    plane.update(bytes(plane.buffer_size))
                fr.sample_rate, fr.pts, fr.time_base = 32000, start, fractions.Fraction(1, 32000)
                for pkt in a.encode(fr):
                    c.mux(pkt)
            for pkt in a.encode():
                c.mux(pkt)
    return str(path)


def test_graph_reference():
    g = workflow.build("P", 768, 1344, 10, 3, 8, "job/r", ref_images=["a.png", "b.png"], ref_videos=["t.mp4"], ref_video_audio=True)
    cond = g["cond"]
    assert cond["class_type"] == "MiniMaxH3ReferenceToVideo"
    i = cond["inputs"]
    assert (i["prompt"], i["width"], i["height"], i["length"]) == ("P", 768, 1344, 243) and i["ref_image_size"] == "match"
    assert i["clip"] == ["clip", 0] and i["vae"] == ["vae_v", 0] and i["audio_vae"] == ["vae_a", 0]
    assert i["ref_images.ref_image_0"] == ["ref_img_0", 0] and g["ref_img_1"]["inputs"]["image"] == "b.png"
    assert g["ref_vid_0"] == {"class_type": "LoadVideo", "inputs": {"file": "t.mp4"}}
    assert i["ref_videos.ref_video_0"] == ["ref_vid_0_parts", 0] and i["ref_video_audios.ref_video_audio_0"] == ["ref_vid_0_parts", 1]
    assert "first_frame" not in i and "img_first" not in g
    # downstream wiring is the shared template's: positive = output 0, AV latent = output 1
    assert g["guider"]["inputs"]["conditioning"] == ["cond", 0] and g["sample"]["inputs"]["latent_image"] == ["cond", 1]
    g = workflow.build("P", 768, 1344, 10, 3, 8, "job/r", ref_videos=["t.mp4"])
    assert not any(k.startswith("ref_video_audios.") for k in g["cond"]["inputs"]) and "ref_img_0" not in g
    with pytest.raises(ValueError):
        workflow.build("P", 768, 1344, 10, 3, 8, "job/r", "f.png", ref_images=["a.png"])
    assert workflow.template()["cond"]["class_type"] == "MiniMaxH3ImageToVideo"


def test_fit_reference_caps_long_edge_only():
    assert workflow.fit_reference(Image.new("RGB", (1200, 1600))).size == (1200, 1600)
    assert workflow.fit_reference(Image.new("RGB", (3000, 4000))).size == (1536, 2048)


def test_parse_refs():
    p = h.parse_input({"request": "P", "skip_rewrite": True, "ref_image_urls": ["https://x/a.png"],
                       "ref_video_urls": ["https://x/t.mp4"], "ref_video_audio": True})
    assert p["ref_image_urls"] == ["https://x/a.png"] and p["ref_video_urls"] == ["https://x/t.mp4"] and p["ref_video_audio"] is True
    p = h.parse_input({"request": "P"})
    assert p["ref_image_urls"] == [] and p["ref_video_urls"] == [] and p["ref_video_audio"] is False


@pytest.mark.parametrize("extra", [
    {"ref_image_urls": ["https://x/a.png"]},                                                   # Ref2VA needs skip_rewrite
    {"skip_rewrite": True, "ref_image_urls": "https://x/a.png"},                               # not a list
    {"skip_rewrite": True, "ref_image_urls": ["https://x/a.png", ""]},
    {"skip_rewrite": True, "ref_image_urls": [f"https://x/{i}.png" for i in range(10)]},       # > 9
    {"skip_rewrite": True, "ref_video_urls": [f"https://x/{i}.mp4" for i in range(4)]},        # > 3
    {"skip_rewrite": True, "ref_video_audio": True},                                          # audio without a video
    {"skip_rewrite": True, "ref_video_urls": ["https://x/t.mp4"], "ref_video_audio": "yes"},
    {"skip_rewrite": True, "ref_image_urls": ["https://x/a.png"], "first_frame_url": "https://x/f.png"},
])
def test_parse_refs_rejects(extra):
    with pytest.raises(storage.BadInput):
        h.parse_input({"request": "P", **extra})


def test_canvas_auto_follows_reference_ratio():
    p = h.parse_input({"request": "x"})
    assert h.canvas(p, None, None, 9 / 16) == (768, 1344)
    assert h.canvas(p, Image.new("RGB", (1600, 900)), None, 9 / 16) == (1344, 768)   # a keyframe still wins
    assert h.canvas(h.parse_input({"request": "x", "aspect": "1:1"}), None, None, 9 / 16) == (768, 768)


def test_probe_video(tmp_path):
    info = storage.probe_video("v", make_video(tmp_path / "a.mp4"))
    assert (info["width"], info["height"], info["fps"], info["audio"]) == (96, 160, 24.0, False) and 2.4 <= info["seconds"] <= 2.6
    assert storage.probe_video("v", make_video(tmp_path / "b.mp4", audio=True), need_audio=True)["audio"] is True
    for path, kw in ((make_video(tmp_path / "c.mp4", fps=30), {}),                 # wrong frame rate
                     (make_video(tmp_path / "d.mp4", seconds=1), {}),              # too short
                     (str(tmp_path / "a.mp4"), {"need_audio": True})):              # no soundtrack to use
        with pytest.raises(storage.BadInput):
            storage.probe_video("v", path, **kw)
    (tmp_path / "e.mp4").write_bytes(b"\0" * 2048)
    with pytest.raises(storage.BadInput):
        storage.probe_video("v", str(tmp_path / "e.mp4"))


def test_fetch_video_rejects_before_network(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAME_URL_ALLOW", "frames.example.com")
    monkeypatch.setattr(storage.requests, "get", lambda *a, **k: pytest.fail("must not fetch"))
    with pytest.raises(storage.BadInput):
        storage.fetch_video("ref_video_urls[0]", "https://evil.com/t.mp4", str(tmp_path / "t.mp4"))


def test_handler_ref2v(stubbed, monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "fetch_image", lambda field, url: Image.new("RGB", (3000, 4000), (9, 9, 9)))
    fetched = []

    def fetch_video(field, url, dest, need_audio=False):
        fetched.append((field, url, dest, need_audio))
        open(dest, "wb").write(b"mp4")
        return {"width": 432, "height": 768, "fps": 24.0, "seconds": 10.13, "audio": True}
    monkeypatch.setattr(storage, "fetch_video", fetch_video)
    monkeypatch.setattr(rewrite, "rewrite", lambda *a, **k: pytest.fail("Ref2VA must not rewrite"))
    out = h.handler({"id": "r1", "input": {"request": "SIX SECTIONS", "skip_rewrite": True, "seconds": 10,
                                           "ref_image_urls": ["https://x/lead.jpg", "https://x/sax.jpg"],
                                           "ref_video_urls": ["https://x/template.mp4"], "ref_video_audio": True}})
    assert "error" not in out, out
    assert out["mode"] == "ref2v" and (out["width"], out["height"]) == (768, 1344)   # "auto" follows the template video
    assert out["refs"] == {"images": [(1536, 2048), (1536, 2048)], "video_audio": True,
                           "videos": [{"width": 432, "height": 768, "fps": 24.0, "seconds": 10.13, "audio": True}]}
    assert fetched[0][0] == "ref_video_urls[0]" and fetched[0][2].endswith("r1_ref_video_0.mp4") and fetched[0][3] is True
    g = stubbed["graph"]
    assert g["cond"]["class_type"] == "MiniMaxH3ReferenceToVideo" and g["cond"]["inputs"]["prompt"] == "SIX SECTIONS"
    assert g["ref_img_1"]["inputs"]["image"] == "r1_ref_image_1.png" and g["ref_vid_0"]["inputs"]["file"] == "r1_ref_video_0.mp4"
    assert not list(tmp_path.glob("r1_*"))                   # inputs are cleaned up after the job


def test_handler_ref2v_bad_video_is_bad_input(stubbed, monkeypatch):
    def fetch_video(field, url, dest, need_audio=False):
        raise storage.BadInput(f"{field} 帧率 30.000 不是 24 fps")
    monkeypatch.setattr(storage, "fetch_video", fetch_video)
    out = h.handler({"id": "r2", "input": {"request": "P", "skip_rewrite": True, "ref_video_urls": ["https://x/t.mp4"]}})
    assert out["error"].startswith("bad_input: ref_video_urls[0] 帧率")


# ----------------------------------------------------------------------------- ComfyUI liveness (v0.1.2)
class FakeProc:
    def __init__(self, rc=None):
        self.returncode, self.pid = rc, 0

    def poll(self):
        return self.returncode


def _history_ok(pid):
    return {pid: {"status": {"status_str": "success"}, "outputs": {"save": {"images": [{"filename": "job_x.mp4", "subfolder": "job"}]}}}}


def test_run_graph_tolerates_slow_http(monkeypatch):
    import urllib.error
    monkeypatch.setattr(h, "_procs", {"comfy": FakeProc()})
    monkeypatch.setattr(h.time, "sleep", lambda s: None)
    monkeypatch.setattr(h, "_progress", lambda job, msg: None)
    calls = {"n": 0}

    def comfy(path, body=None, timeout=60):
        if path == "/prompt":
            return {"prompt_id": "p1"}
        calls["n"] += 1
        if calls["n"] < 4:
            raise urllib.error.URLError(TimeoutError("timed out"))   # busy server, process alive
        return _history_ok("p1")
    monkeypatch.setattr(h, "comfy", comfy)
    path, _ = h.run_graph({}, {}, h.time.time() + 600)
    assert path.endswith("job/job_x.mp4") and calls["n"] == 4


def test_run_graph_sigkill_is_oom(monkeypatch):
    proc = FakeProc()
    monkeypatch.setattr(h, "_procs", {"comfy": proc})
    monkeypatch.setattr(h.time, "sleep", lambda s: None)
    monkeypatch.setattr(h, "comfy_death_report", lambda: "exit=-9; oom=1 oom_kill=1")

    def comfy(path, body=None, timeout=60):
        if path == "/prompt":
            return {"prompt_id": "p1"}
        proc.returncode = -9
        return {}
    monkeypatch.setattr(h, "comfy", comfy)
    with pytest.raises(h.JobError) as e:
        h.run_graph({}, {}, h.time.time() + 600)
    assert e.value.code == "oom" and e.value.refresh and "exit=-9" in str(e.value)


def test_run_graph_live_but_silent_gives_up(monkeypatch):
    import urllib.error
    monkeypatch.setattr(h, "_procs", {"comfy": FakeProc()})
    monkeypatch.setattr(h, "COMFY_SILENT_S", 0.0)
    monkeypatch.setattr(h.time, "sleep", lambda s: None)

    def comfy(path, body=None, timeout=60):
        if path == "/prompt":
            return {"prompt_id": "p1"}
        raise urllib.error.URLError(TimeoutError("timed out"))
    monkeypatch.setattr(h, "comfy", comfy)
    with pytest.raises(h.JobError, match="unresponsive") as e:
        h.run_graph({}, {}, h.time.time() + 600)
    assert e.value.code == "internal"


def test_comfy_alive_is_process_based(monkeypatch):
    monkeypatch.setattr(h, "comfy", lambda *a, **k: pytest.fail("liveness must not depend on HTTP"))
    monkeypatch.setattr(h, "_procs", {"comfy": FakeProc()})
    assert h.comfy_alive()
    monkeypatch.setattr(h, "_procs", {"comfy": FakeProc(-9)})
    assert not h.comfy_alive()


# ----------------------------------------------------------------------------- memory sampler (v0.1.3)
def test_mem_report_phases_and_events():
    m = h.MemSampler()
    m.samples.extend([(1.0, "boot", 100, 5000, 5100), (5.0, "rewrite+boot", 900, 20000, 20900),
                      (9.0, "generating", 30000, 12000, 42000), (13.0, "generating", 38000, 6000, 44000), (17.0, "uploading", 20000, 9000, 29000)])
    m.events.extend([(2.0, "prefetch start"), (10.0, "Requested to load MiniMaxH3"), (12.0, "loaded partially 18000 MB")])
    r = m.report(since=4.0)
    assert set(r["phases"]) == {"rewrite+boot", "generating", "uploading"}
    assert r["phases"]["generating"]["anon_max"] == 38000 and r["phases"]["generating"]["from"] == 9.0
    assert [e[1] for e in r["events"]] == ["Requested to load MiniMaxH3", "loaded partially 18000 MB"]
    assert r["timeline"][0] == [5.0, "rewrite+boot", 900, 20000]


def test_mem_event_regex():
    for line in ("Requested to load MiniMaxH3", "loaded partially; 18000.00 MB usable", "Prompt executed in 170.1 seconds",
                 "torch.OutOfMemoryError: CUDA out of memory"):
        assert h.MemSampler.EVENT_RE.search(line), line
    assert not h.MemSampler.EVENT_RE.search(" 50%|█████     | 4/8 [01:10<01:10, 17.5s/it]")


def test_memstat_shape():
    m = h.memstat()          # whatever this machine exposes; must not raise
    assert m is None or {"current", "anon", "max"} <= set(m)
