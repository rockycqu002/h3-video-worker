"""RunPod Serverless handler: MiniMax H3 (DaSiWa Hybrid 8-step) text/keyframe -> video+audio on one RTX 4090 via ComfyUI.

Protocol (see docs/PLAN.md §3):
  input.request          plain request (or the final H3 prompt when skip_rewrite=true)   required
  input.skip_rewrite     bool, default false
  input.creativity       restrained | balanced | bold | extreme, default balanced
  input.seconds          4–15, default 5
  input.aspect           auto | 21:9 | 16:9 | 4:3 | 1:1 | 3:4 | 9:16, default auto
  input.quality          768 | 640 | 544 | 416 (short edge), default 768
  input.seed / steps     optional
  input.first_frame_url / last_frame_url   https URLs on an allow-listed host (R2 presigned)
returns {"video_key": "videos/<job_id>.mp4", ...} or {"error": "<code>: <message>"} (RunPod reports it as FAILED).

At boot ComfyUI and open-h3-ir start in the background and the weight files are read into the page cache, so the
first job's rewrite overlaps with the cold start. Models are loaded by ComfyUI on the first prompt and then reused.
"""
import atexit, collections, concurrent.futures, glob, json, os, re, secrets, shlex, signal, subprocess, sys, threading, time, traceback, urllib.error, urllib.request

os.environ.setdefault("RUNPOD_LOG_LEVEL", "WARN")   # the SDK would otherwise log job payloads at DEBUG/INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rewrite, storage, workflow  # noqa: E402
from storage import BadInput  # noqa: E402

COMFY_DIR = os.environ.get("COMFY_DIR", "/app/ComfyUI")
COMFY_URL = "http://127.0.0.1:8188"
IN_DIR, OUT_DIR, TMP_DIR = "/tmp/comfy_in", "/tmp/comfy_out", "/tmp/comfy_tmp"
MODELS_DIR = os.environ.get("H3_MODELS_DIR", "/runpod-volume/h3/models")
# load order in the graph: text encoder first, then the DiT, then the VAEs
MODEL_FILES = ["text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
               "diffusion_models/DasiwaMinimaxH3_dasiwaHybrid8turboV1.safetensors",
               "vae/minimax_h3_video_vae_fp16.safetensors",
               "vae/minimax_h3_audio_vae_fp32.safetensors"]
H3IR_BIN = os.environ.get("H3IR_BIN", "/opt/h3ir/bin/h3ir")
COMFY_EXTRA_ARGS = shlex.split(os.environ.get("H3_COMFY_ARGS", "--reserve-vram 1 --disable-nvml-pressure"))
BUILD = os.environ.get("H3_BUILD", "dev")
JOB_DEADLINE_S = float(os.environ.get("H3_JOB_DEADLINE_S", 840))     # stay under the endpoint executionTimeout (900 s)
COMFY_BOOT_S = float(os.environ.get("H3_COMFY_BOOT_S", 300))
COMFY_SILENT_S = float(os.environ.get("H3_COMFY_SILENT_S", 180))  # tolerated HTTP silence from a live ComfyUI mid-generation
PREFETCH = os.environ.get("H3_PREFETCH", "1") == "1"
MAX_REQUEST_CHARS = 6000
DEFAULTS = {"seconds": 5.0, "aspect": "auto", "quality": "768", "steps": 8, "creativity": "balanced"}
LIMITS = {"seconds": (4.0, 15.0), "steps": (6, 12), "seed": (0, 2**53)}

_procs = {}
_state = {"boot_t0": time.time(), "comfy_ready_s": None, "jobs": 0, "models_missing": None, "prefetch_s": None}


# ----------------------------------------------------------------------------- logging / child processes
def log(msg, **kv):
    extra = (" " + json.dumps(kv, ensure_ascii=False, default=str)) if kv else ""
    print(f"[h3 {time.strftime('%H:%M:%S')}] {msg}{extra}", flush=True)


class JobError(Exception):
    def __init__(self, code, message, refresh=False):
        super().__init__(message); self.code, self.refresh = code, refresh


def _tail(path, tag):
    """Forward a child's log file to stdout so RunPod captures it; never let the child block on a pipe."""
    with open(path, "r", errors="replace") as f:
        while True:
            line = f.readline()
            if line:
                sys.stdout.write(f"[{tag}] " + line); sys.stdout.flush()
                if tag == "comfy" and MEM.EVENT_RE.search(line):
                    MEM.event(line)
            else:
                time.sleep(0.2)


def _spawn(name, cmd, cwd=None, env=None):
    path = f"/tmp/{name}.log"
    fh = open(path, "a")
    _procs[name] = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    threading.Thread(target=_tail, args=(path, name), daemon=True).start()


def start_comfy():
    for d in (IN_DIR, OUT_DIR, TMP_DIR):
        os.makedirs(d, exist_ok=True)
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch", "--dont-print-server",
           "--disable-metadata", "--input-directory", IN_DIR, "--output-directory", OUT_DIR, "--temp-directory", TMP_DIR,
           "--extra-model-paths-config", os.path.join(COMFY_DIR, "extra_model_paths.yaml")] + COMFY_EXTRA_ARGS
    _spawn("comfy", cmd, cwd=COMFY_DIR)


def start_h3ir():
    if not os.path.exists(H3IR_BIN):
        log("h3ir binary missing; rewrites will use the skill fallback", path=H3IR_BIN); return
    env = dict(os.environ, H3IR_LLM_URL=os.environ.get("H3IR_LLM_URL", "https://openrouter.ai/api/v1"),
               H3IR_LLM_MODEL=rewrite.OPENROUTER_MODEL, H3IR_LLM_KEY=os.environ.get("OPENROUTER_API_KEY", ""),
               H3IR_COMFY_URL=COMFY_URL, H3IR_STATE_DIR="/tmp/h3ir")
    os.makedirs("/tmp/h3ir", exist_ok=True)
    _spawn("h3ir", [H3IR_BIN, "serve", "--port", rewrite.H3IR_URL.rsplit(":", 1)[1]], env=env)


def stop_children(*_):
    for p in _procs.values():
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(timeout=10)
            except Exception:
                try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception: pass


def check_models():
    missing = [f for f in MODEL_FILES if not os.path.isfile(os.path.join(MODELS_DIR, f))]
    _state["models_missing"] = missing
    if missing:
        log("model files missing on the volume", models_dir=MODELS_DIR, missing=missing)
    return not missing


def prefetch():
    """Read the weights once so ComfyUI's first load comes from the page cache instead of the network volume."""
    t0 = time.time()
    MEM.event("prefetch start")
    for rel in MODEL_FILES:
        try:
            with open(os.path.join(MODELS_DIR, rel), "rb", buffering=0) as f:
                while f.read(64 << 20):
                    pass
        except OSError as e:
            log("prefetch failed", file=rel, error=str(e)); return
    _state["prefetch_s"] = round(time.time() - t0, 1)
    MEM.event("prefetch done")
    log("prefetch done", seconds=_state["prefetch_s"])


# ----------------------------------------------------------------------------- ComfyUI HTTP
def comfy(path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(COMFY_URL + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"error": raw[:600].decode(errors="replace")}
        raise JobError("comfy_rejected", f"ComfyUI HTTP {e.code}: " + json.dumps(payload.get("node_errors") or payload.get("error") or payload, ensure_ascii=False)[:700])
    return json.loads(raw) if raw.strip() else {}


def comfy_alive():
    """Process liveness only. Under GPU/RAM pressure ComfyUI's HTTP server can take many seconds to answer, so a slow
    /system_stats must never be taken as "dead" (v0.1.0 failed a healthy job that way)."""
    p = _procs.get("comfy")
    return p is not None and p.poll() is None


def comfy_death_report():
    """Why ComfyUI is gone: exit code (-9 = SIGKILL, usually the OOM killer), cgroup OOM counters, memory, last log lines."""
    p = _procs.get("comfy")
    parts = [f"exit={p.returncode if p else None}"]
    try:
        ev = dict(line.split()[:2] for line in open("/sys/fs/cgroup/memory.events"))
        parts.append(f"oom={ev.get('oom')} oom_kill={ev.get('oom_kill')}")
    except (OSError, ValueError):
        pass
    mem = container_mem_mib() or {}
    parts.append(f"mem_peak={mem.get('peak')}MiB max={mem.get('max')}MiB anon={mem.get('anon')}MiB file={mem.get('file')}MiB")
    try:
        r = MEM.report(since=_state.get("job_t", 0.0))
        parts.append("anon_max_by_phase=" + ",".join(f"{k}:{v['anon_max']}" for k, v in r["phases"].items()))
        parts.append("last_events=" + " | ".join(f"{t}s {e[:80]}" for t, e in r["events"][-4:]))
    except Exception:
        pass
    try:
        tail = [l.strip() for l in open("/tmp/comfy.log", errors="replace").readlines()[-40:] if l.strip()]
        keep = [l for l in tail if any(k in l for k in ("Error", "error", "Killed", "memory", "Traceback", "CUDA"))][-3:] or tail[-2:]
        parts.append("log: " + " | ".join(x[:200] for x in keep))
    except OSError:
        pass
    return "; ".join(parts)


def wait_comfy(deadline):
    while time.time() < deadline:
        p = _procs.get("comfy")
        if p is None or p.poll() is not None:
            raise JobError("internal", f"ComfyUI exited during start ({comfy_death_report()})", refresh=True)
        try:
            comfy("/system_stats", timeout=5)
            if _state["comfy_ready_s"] is None:
                _state["comfy_ready_s"] = round(time.time() - _state["boot_t0"], 1)
                log("comfyui up", seconds_since_boot=_state["comfy_ready_s"])
            return
        except Exception:
            time.sleep(1)
    raise JobError("internal", "ComfyUI did not come up in time", refresh=True)


def vram_used_mib():
    try:
        d = comfy("/system_stats", timeout=5)["devices"][0]
        return (d["vram_total"] - d["vram_free"]) // 2**20
    except Exception:
        return None


def _read_int(path):
    try:
        v = open(path).read().strip()
        return None if v == "max" else int(v)
    except (OSError, ValueError):
        return None


def _read_kv(path):
    try:
        return {k: int(v) for k, v in (line.split()[:2] for line in open(path))}
    except (OSError, ValueError):
        return {}


def memstat():
    """The container's own memory in MiB (cgroup v2, falling back to v1); /proc/meminfo would report the whole host.
    anon = process memory (what the OOM killer is about); file = page cache (prefetched / mmapped weights, reclaimable)."""
    mib = lambda b: None if b is None else b // 2**20
    if os.path.exists("/sys/fs/cgroup/memory.current"):
        st, ev = _read_kv("/sys/fs/cgroup/memory.stat"), _read_kv("/sys/fs/cgroup/memory.events")
        return {"cg": 2, "current": mib(_read_int("/sys/fs/cgroup/memory.current")), "peak": mib(_read_int("/sys/fs/cgroup/memory.peak")),
                "max": mib(_read_int("/sys/fs/cgroup/memory.max")), "anon": mib(st.get("anon")), "file": mib(st.get("file")),
                "swap": mib(_read_int("/sys/fs/cgroup/memory.swap.current")), "oom_kill": ev.get("oom_kill"), "high_events": ev.get("high")}
    base = "/sys/fs/cgroup/memory"
    if os.path.exists(f"{base}/memory.usage_in_bytes"):
        st = _read_kv(f"{base}/memory.stat")
        limit = _read_int(f"{base}/memory.limit_in_bytes")
        oom = _read_kv(f"{base}/memory.oom_control")
        return {"cg": 1, "current": mib(_read_int(f"{base}/memory.usage_in_bytes")), "peak": mib(_read_int(f"{base}/memory.max_usage_in_bytes")),
                "max": mib(limit) if limit and limit < 2**60 else None, "anon": mib(st.get("total_rss", st.get("rss"))),
                "file": mib(st.get("total_cache", st.get("cache"))), "swap": mib(st.get("total_swap")), "oom_kill": oom.get("oom_kill")}
    return None


def container_mem_mib():
    m = memstat()
    return {k: m[k] for k in ("current", "peak", "max", "anon", "file", "oom_kill") if k in m} if m else None


class MemSampler:
    """Samples memstat() every 2 s from boot, tagged with the handler's current phase; ComfyUI log lines about model
    loading are timestamped by the log forwarder. Together they show *when* memory grows (load / sample / decode)."""
    EVENT_RE = re.compile(r"(Requested to load|loaded (completely|partially)|[Uu]nload|Prompt executed|out of memory|OutOfMemory|Killed|"
                          r"model_type|Using .* attention|lowvram|offload|Traceback|Error)")

    def __init__(self, interval=2.0):
        self.t0, self.interval, self.phase = time.time(), interval, "boot"
        self.samples = collections.deque(maxlen=4000)      # (t, phase, anon, file, current)
        self.events = collections.deque(maxlen=300)        # (t, text)
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def set_phase(self, phase):
        self.phase = phase

    def event(self, text):
        with self.lock:
            self.events.append((round(time.time() - self.t0, 1), text.strip()[:180]))

    def _run(self):
        last_log = 0
        while True:
            m = memstat() or {}
            t = round(time.time() - self.t0, 1)
            with self.lock:
                self.samples.append((t, self.phase, m.get("anon"), m.get("file"), m.get("current")))
            if t - last_log >= 10:                          # also into the RunPod log, in case the worker dies
                log("mem", t=t, phase=self.phase, anon=m.get("anon"), file=m.get("file"), current=m.get("current"), max=m.get("max"),
                    oom_kill=m.get("oom_kill"))
                last_log = t
            time.sleep(self.interval)

    def report(self, since=0.0, points=60):
        """Per-phase peaks + a downsampled timeline + ComfyUI events, all from `since` (seconds after boot)."""
        with self.lock:
            s = [x for x in self.samples if x[0] >= since]
            ev = [e for e in self.events if e[0] >= since]
        phases = {}
        for t, ph, anon, file, cur in s:
            p = phases.setdefault(ph, {"anon_max": 0, "current_max": 0, "from": t, "to": t})
            p["anon_max"] = max(p["anon_max"], anon or 0); p["current_max"] = max(p["current_max"], cur or 0); p["to"] = t
        step = max(1, len(s) // points)
        return {"limit": (memstat() or {}).get("max"), "phases": phases,
                "timeline": [[t, ph, anon, file] for t, ph, anon, file, _ in s[::step]], "events": ev[-40:]}


MEM = MemSampler()


# ----------------------------------------------------------------------------- input
def _number(v, name, default, cast):
    if v is None or v == "":
        return default
    if isinstance(v, bool):
        raise BadInput(f"{name} 不是数字")
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise BadInput(f"{name} 不是数字")
    lo, hi = LIMITS[name]
    if not (lo <= v <= hi):
        raise BadInput(f"{name}={v} 超出允许范围 {lo}–{hi}")
    return v


def _choice(v, name, options, default):
    if v is None or v == "":
        return default
    v = str(v)
    if v not in options:
        raise BadInput(f"{name}={v} 不合法，可选: {', '.join(options)}")
    return v


def parse_input(inp):
    if not isinstance(inp, dict):
        raise BadInput("input 必须是对象")
    req = inp.get("request")
    if not isinstance(req, str) or not req.strip():
        raise BadInput("request 缺失")
    if len(req) > MAX_REQUEST_CHARS:
        raise BadInput(f"request 超过 {MAX_REQUEST_CHARS} 字符")
    skip = inp.get("skip_rewrite", False)
    if not isinstance(skip, bool):
        raise BadInput("skip_rewrite 必须是布尔值")
    seed = _number(inp.get("seed"), "seed", None, int)
    return {
        "request": req.strip(), "skip_rewrite": skip,
        "creativity": _choice(inp.get("creativity"), "creativity", rewrite.CREATIVITY, DEFAULTS["creativity"]),
        "seconds": _number(inp.get("seconds"), "seconds", DEFAULTS["seconds"], float),
        "aspect": _choice(inp.get("aspect"), "aspect", ["auto"] + list(workflow.ASPECTS), DEFAULTS["aspect"]),
        "quality": _choice(inp.get("quality"), "quality", list(workflow.QUALITIES), DEFAULTS["quality"]),
        "steps": _number(inp.get("steps"), "steps", DEFAULTS["steps"], int),
        "seed": seed if seed is not None else secrets.randbelow(2**32),
        "first_frame_url": inp.get("first_frame_url") or None, "last_frame_url": inp.get("last_frame_url") or None,
    }


def canvas(p, first, last):
    """"auto" follows the keyframe's aspect (first frame preferred), otherwise 16:9."""
    anchor = first if first is not None else last
    if p["aspect"] == "auto":
        if anchor is None:
            return workflow.resolution("16:9", p["quality"])
        return workflow.resolution("16:9", p["quality"], ratio=anchor.width / anchor.height)
    return workflow.resolution(p["aspect"], p["quality"])


# ----------------------------------------------------------------------------- generation
def _progress(job, msg):
    try:
        import runpod
        runpod.serverless.progress_update(job, msg)
    except Exception:
        pass


def execution_error(status):
    for m in status.get("messages", []):
        if m[0] == "execution_error":
            d = m[1]
            text = f"{d.get('node_type')}: {d.get('exception_type', '')} {d.get('exception_message', '')}".strip()
            if "OutOfMemory" in text or "out of memory" in text.lower():
                return JobError("oom", text[:600], refresh=True)
            return JobError("comfy_rejected", text[:600])
    return JobError("comfy_rejected", "ComfyUI execution failed: " + json.dumps(status, ensure_ascii=False)[:500])


def run_graph(job, graph, deadline):
    pid = comfy("/prompt", {"prompt": graph, "client_id": "h3-worker"}).get("prompt_id")
    if not pid:
        raise JobError("comfy_rejected", "ComfyUI returned no prompt_id")
    t0, last_note, last_ok = time.time(), 0, time.time()
    while True:
        try:
            h = comfy(f"/history/{pid}", timeout=30)
            last_ok = time.time()
            if pid in h:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            # a busy ComfyUI answers slowly; only give up if the process is gone or it stays silent for long
            if comfy_alive() and time.time() - last_ok > COMFY_SILENT_S:
                raise JobError("internal", f"ComfyUI alive but unresponsive for {time.time() - last_ok:.0f} s ({type(e).__name__})", refresh=True)
        if not comfy_alive():
            report = comfy_death_report()
            killed = "exit=-9" in report            # SIGKILL: in a container this is almost always the OOM killer
            raise JobError("oom" if killed else "internal", f"ComfyUI died during generation ({report})", refresh=True)
        if time.time() > deadline:
            try:
                comfy("/queue", {"delete": [pid]}); comfy("/interrupt", {})
            except Exception:
                pass
            raise JobError("generate_timeout", f"generation exceeded the job deadline after {time.time() - t0:.0f} s")
        if time.time() - last_note > 15:
            _progress(job, f"generating {time.time() - t0:.0f}s"); last_note = time.time()
        time.sleep(1)
    st = h[pid].get("status", {})
    if st.get("status_str") != "success":
        raise execution_error(st)
    for node in h[pid].get("outputs", {}).values():
        for item in node.get("images", []) + node.get("gifs", []) + node.get("videos", []):
            if item.get("filename", "").endswith(".mp4"):
                return os.path.join(OUT_DIR, item.get("subfolder", ""), item["filename"]), time.time() - t0
    raise JobError("comfy_rejected", "ComfyUI produced no mp4")


def _cleanup(paths):
    for p in paths:
        try: os.remove(p)
        except OSError: pass


def handler(job):
    jid = str(job.get("id") or "")
    t0 = time.time()
    _state["job_t"] = round(t0 - MEM.t0, 1)
    MEM.set_phase("preparing"); MEM.event(f"job start {jid}")
    timing, paths = {"boot": None}, []
    try:
        try:
            key = storage.video_key(jid)
        except ValueError as e:
            raise JobError("internal", str(e))
        p = parse_input(job.get("input"))
        if _state["models_missing"]:
            raise JobError("internal", f"model files missing on the volume: {_state['models_missing']}")
        deadline = t0 + JOB_DEADLINE_S
        _progress(job, "preparing")
        first = storage.fetch_image("first_frame_url", p["first_frame_url"]) if p["first_frame_url"] else None
        last = storage.fetch_image("last_frame_url", p["last_frame_url"]) if p["last_frame_url"] else None
        first = workflow.normalize(first) if first is not None else None
        last = workflow.normalize(last) if last is not None else None
        w, h = canvas(p, first, last)
        first = workflow.cover_crop(first, w, h) if first is not None else None
        last = workflow.cover_crop(last, w, h) if last is not None else None
        frames = workflow.frames_for_seconds(p["seconds"])

        # rewrite runs in a thread while this thread waits for ComfyUI (cold start) — the two overlap
        fut = None
        if not p["skip_rewrite"]:
            _progress(job, "rewriting"); MEM.set_phase("rewrite+boot")
            ex = concurrent.futures.ThreadPoolExecutor(1)
            fut = ex.submit(rewrite.rewrite, p["request"], p["seconds"], p["aspect"], frames, workflow.FPS, first, last,
                            p["creativity"], os.environ.get("OPENROUTER_API_KEY", ""), TMP_DIR)
            ex.shutdown(wait=False)
        names = {}
        for tag, im in (("first", first), ("last", last)):
            if im is not None:
                names[tag] = f"{jid}_{tag}.png"
                path = os.path.join(IN_DIR, names[tag]); im.save(path, format="PNG", compress_level=1); paths.append(path)
        tb = time.time()
        if fut is None:
            MEM.set_phase("boot-wait")
        wait_comfy(min(deadline, time.time() + COMFY_BOOT_S))
        timing["boot"] = round(time.time() - tb, 1)
        if fut is not None:
            try:
                rw = fut.result(timeout=max(1, deadline - time.time()))
            except rewrite.RewriteError as e:
                raise JobError("rewrite_failed", str(e))
            except concurrent.futures.TimeoutError:
                raise JobError("rewrite_failed", "rewrite exceeded the job deadline")
            prompt = rw.pop("prompt")
        else:
            prompt, rw = p["request"], {"engine": "none", "status": "skipped", "seconds": 0}
        timing["rewrite"] = rw.get("seconds")

        _progress(job, "generating"); MEM.set_phase("generating")
        graph = workflow.build(prompt, w, h, p["seconds"], p["seed"], p["steps"], f"job/{jid}", names.get("first"), names.get("last"))
        out_path, gen_s = run_graph(job, graph, deadline)
        paths.append(out_path)
        timing["generate"] = round(gen_s, 1)

        _progress(job, "uploading"); MEM.set_phase("uploading")
        tu = time.time()
        try:
            nbytes = storage.upload_video(out_path, key)
        except Exception as e:
            raise JobError("upload_failed", f"{type(e).__name__}: {str(e)[:300]}")
        timing["upload"] = round(time.time() - tu, 1)
        timing["total"] = round(time.time() - t0, 1)
        _state["jobs"] += 1
        out = {"video_key": key, "bytes": nbytes, "width": w, "height": h, "frames": frames, "fps": workflow.FPS,
               "seconds": p["seconds"], "seed": p["seed"], "steps": p["steps"], "mode": rewrite.mode_of(first, last),
               "prompt": prompt, "rewrite": rw, "timing": timing, "build": BUILD,
               "worker": {"jobs": _state["jobs"], "comfy_ready_s": _state["comfy_ready_s"], "prefetch_s": _state["prefetch_s"],
                          "vram_used_mib": vram_used_mib(), "mem_mib": container_mem_mib(), "mem": MEM.report(since=_state["job_t"])}}
        log("job ok", job=jid, **{k: v for k, v in out.items() if k != "prompt"})
        return out
    except BadInput as e:
        log("job rejected", job=jid, error=str(e))
        return {"error": f"bad_input: {e}"}
    except JobError as e:
        log("job failed", job=jid, code=e.code, error=str(e))
        refresh = e.refresh or not comfy_alive()
        return {"error": f"{e.code}: {e}", **({"refresh_worker": True} if refresh else {})}
    except Exception as e:
        log("job failed", job=jid, code="internal", error=str(e)[:800], tb=traceback.format_exc()[-1500:])
        return {"error": f"internal: {type(e).__name__}: {str(e)[:500]}", **({"refresh_worker": True} if not comfy_alive() else {})}
    finally:
        MEM.set_phase("idle"); MEM.event(f"job end {jid}")
        _cleanup(paths + glob.glob(os.path.join(OUT_DIR, "job", f"{jid}*")))


def main():
    atexit.register(stop_children)
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: (stop_children(), sys.exit(0)))
    MEM.start()
    log("boot", build=BUILD, comfy_args=" ".join(COMFY_EXTRA_ARGS), models_dir=MODELS_DIR)
    start_comfy()
    start_h3ir()
    if check_models() and PREFETCH:
        threading.Thread(target=prefetch, daemon=True).start()
    import runpod
    runpod.serverless.start({"handler": handler})   # take jobs right away: the first job's rewrite overlaps the cold start


if __name__ == "__main__":
    main()
