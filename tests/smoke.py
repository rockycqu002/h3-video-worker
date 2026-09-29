#!/usr/bin/env python3
"""Live smoke test against a RunPod endpoint: submit via /run, poll /status, print the result and wall-clock timing.

  python tests/smoke.py --endpoint <id> t2v|i2v|fl2v|skip [--seconds 5] [--frames-base https://raw.githubusercontent.com/<owner>/<repo>/<sha>/tests/data]

The API key comes from $RUNPOD_API_KEY or ~/.config/runpod/api_key and is never printed.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

API = "https://api.runpod.ai/v2"
TERMINAL = ("COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED")
CASES = {
    "t2v": {"request": "A barista in a sunlit café slowly pours steamed milk into a cup, forming latte art; soft espresso-machine hiss, "
                       "cups clinking, quiet indie guitar music."},
    "i2v": {"request": "The barista finishes pouring the latte art, looks up and smiles at the customer, then slides the cup across the counter."},
    "fl2v": {"request": "The barista finishes the latte art and sets the cup down on the wooden counter; the camera tilts down to the cup."},
    "skip": {"skip_rewrite": True,
             "request": "integrated_multimodal_description: [Shot 1] A close-up of a white ceramic cup on a wooden café counter; steam rises "
                        "slowly while morning light moves across the grain.\n\noverall_soundscape: soft café ambience, a distant espresso "
                        "machine hiss.\n\nnon_diegetic_music: gentle acoustic guitar."},
}


def api_key():
    return os.environ.get("RUNPOD_API_KEY") or open(os.path.expanduser("~/.config/runpod/api_key")).read().strip()


def req(method, url, body=None, timeout=30):
    r = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method,
                               headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json", "User-Agent": "h3-video-worker-tests/1.0"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True); ap.add_argument("case", choices=list(CASES))
    ap.add_argument("--seconds", type=float, default=5); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames-base"); ap.add_argument("--timeout", type=float, default=1200)
    a = ap.parse_args()
    inp = {**CASES[a.case], "seconds": a.seconds, "seed": a.seed}
    if a.case in ("i2v", "fl2v"):
        if not a.frames_base:
            sys.exit("--frames-base is required for keyframe cases")
        inp["first_frame_url"] = f"{a.frames_base}/first.jpg"
        if a.case == "fl2v":
            inp["last_frame_url"] = f"{a.frames_base}/last.jpg"
    t0 = time.time()
    job = req("POST", f"{API}/{a.endpoint}/run", {"input": inp})
    jid = job["id"]; print(f"[{a.case}] job {jid} submitted", flush=True)
    last = None
    while time.time() - t0 < a.timeout:
        try:
            st = req("GET", f"{API}/{a.endpoint}/status/{jid}")
        except (urllib.error.URLError, TimeoutError) as e:
            print("  poll error", type(e).__name__, flush=True); time.sleep(5); continue
        note = (st.get("status"), st.get("output") if st.get("status") == "IN_PROGRESS" and isinstance(st.get("output"), str) else None)
        if note != last:
            print(f"  {time.time() - t0:6.0f}s {note[0]} {note[1] or ''}", flush=True); last = note
        if st.get("status") in TERMINAL:
            out = st.get("output")
            if isinstance(out, dict) and "prompt" in out:
                out = {**out, "prompt": out["prompt"][:400] + ("…" if len(out["prompt"]) > 400 else "")}
            print(json.dumps({"status": st["status"], "wall_s": round(time.time() - t0, 1), "delayTime_ms": st.get("delayTime"),
                              "executionTime_ms": st.get("executionTime"), "error": st.get("error"), "output": out}, ensure_ascii=False, indent=1))
            sys.exit(0 if st["status"] == "COMPLETED" else 1)
        time.sleep(5)
    sys.exit("client timeout")


if __name__ == "__main__":
    main()
