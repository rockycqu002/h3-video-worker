#!/usr/bin/env python3
"""Create / read / update the RunPod Serverless template + endpoint for the H3 worker through the v1 REST API
(adapted from qwen-image-edit-4090/deploy/runpod_api.py).

  python deploy/runpod_api.py create --image ghcr.io/rockycqu002/h3-video-worker:v0.1.0@sha256:... [--max-workers 1] [--extra-frame-host raw.githubusercontent.com]
  python deploy/runpod_api.py show   --endpoint <id>
  python deploy/runpod_api.py update --endpoint <id> --max-workers 3
  python deploy/runpod_api.py template-update --template <id> --image <ref@digest> [--env KEY=VAL ...]
  python deploy/runpod_api.py billing --endpoint <id> [--days 1]

Secrets are referenced as {{ RUNPOD_SECRET_<name> }} (created by the account owner in the console); their values never pass
through this script. Only the ids you pass are touched; the pre-existing endpoints of the account are refused.
The API key comes from $RUNPOD_API_KEY or ~/.config/runpod/api_key and is never printed.
"""
import argparse, json, os, sys, urllib.error, urllib.request

REST = "https://rest.runpod.io/v1"
GPU = "NVIDIA GeForce RTX 4090"
VOLUMES = {"US-CA-2": "zpzkjk80go"}          # h3-models-us-ca-2, filled by scripts/fetch_models.py
R2_ACCOUNT = "21c38efd61d6ef0a93541eb0cb1bc3fa"
FOREIGN = {"8mnjqb7tixo7hu", "btrcmfepllqgbf", "jlg3cxj1nk7l5a", "n4cjbcgch8r51w", "ngxgcsfy99ohlu", "oruj8lz2g2gqri",
           "pijlsbbbn02s4p", "w2ww53ksg9wtj7"}   # the account's other endpoints: never modify them from here


def key():
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        k = open(os.path.expanduser("~/.config/runpod/api_key")).read().strip()
    return k


def call(method, path, body=None):
    req = urllib.request.Request(REST + path, data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json", "User-Agent": "h3-video-worker-deploy/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} -> HTTP {e.code}: {e.read()[:800].decode(errors='replace')}")


def base_env(extra_hosts=()):
    hosts = [f"{R2_ACCOUNT}.r2.cloudflarestorage.com", *extra_hosts]
    return {
        "OPENROUTER_API_KEY": "{{ RUNPOD_SECRET_h3_openrouter_key }}",
        "OPENROUTER_MODEL": "qwen/qwen3-vl-235b-a22b-instruct",
        "R2_ENDPOINT": f"https://{R2_ACCOUNT}.r2.cloudflarestorage.com",
        "R2_BUCKET": "h3-videos",
        "R2_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_h3_r2_access_key_id }}",
        "R2_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_h3_r2_secret_access_key }}",
        "FRAME_URL_ALLOW": ",".join(hosts),
        "H3_COMFY_ARGS": "--reserve-vram 1 --disable-nvml-pressure --fast-disk",
        "H3_JOB_DEADLINE_S": "1740",
        "H3_PREFETCH": "1",
    }


def guard(endpoint):
    if endpoint.strip() in FOREIGN:
        sys.exit("refusing to modify an endpoint that does not belong to the H3 worker")


def create(a):
    if a.template:
        tpl = call("GET", f"/templates/{a.template}")
    else:
        tpl = call("POST", "/templates", {"name": a.name, "imageName": a.image, "isServerless": True, "containerDiskInGb": a.disk,
                                          "volumeInGb": 0, "ports": [], "env": base_env(a.extra_frame_host or ()), "category": "NVIDIA"})
    print("template:", json.dumps({k: tpl.get(k) for k in ("id", "name", "imageName", "containerDiskInGb")}, ensure_ascii=False))
    ep = call("POST", "/endpoints", {
        "templateId": tpl["id"], "name": a.name, "computeType": "GPU", "gpuTypeIds": [GPU], "gpuCount": 1,
        "minCudaVersion": "13.0", "dataCenterIds": list(VOLUMES),
        # single region: networkVolumeId; multi-region needs networkVolumeIds as a list of objects (schema NetworkVolumeIdsInput)
        "networkVolumeId": next(iter(VOLUMES.values())),
        "workersMin": 0, "workersMax": a.max_workers, "idleTimeout": a.idle, "executionTimeoutMs": 1800000,   # 15 s clips generate for ~17 min
        "flashboot": True, "scalerType": "REQUEST_COUNT", "scalerValue": 1})
    print("endpoint:", ep.get("id"))
    show(argparse.Namespace(endpoint=ep["id"]))


def show(a):
    ep = call("GET", f"/endpoints/{a.endpoint}")
    keep = ("id", "name", "templateId", "gpuTypeIds", "gpuCount", "minCudaVersion", "dataCenterIds", "networkVolumeId", "networkVolumeIds",
            "workersMin", "workersMax", "idleTimeout", "executionTimeoutMs", "scalerType", "scalerValue", "createdAt")
    print(json.dumps({k: ep.get(k) for k in keep if k in ep}, ensure_ascii=False, indent=1))
    if ep.get("templateId"):
        tpl = call("GET", f"/templates/{ep['templateId']}")
        env = {k: (v if "RUNPOD_SECRET" in v or "KEY" not in k else "<set>") for k, v in (tpl.get("env") or {}).items()}
        print("template:", json.dumps({"id": tpl.get("id"), "imageName": tpl.get("imageName"), "containerDiskInGb": tpl.get("containerDiskInGb"), "env": env},
                                      ensure_ascii=False, indent=1))


def update(a):
    guard(a.endpoint)
    body = {}
    if a.max_workers is not None: body["workersMax"] = a.max_workers
    if a.min_workers is not None: body["workersMin"] = a.min_workers
    if a.idle is not None: body["idleTimeout"] = a.idle
    if a.timeout_ms is not None: body["executionTimeoutMs"] = a.timeout_ms
    if not body:
        sys.exit("nothing to update (an empty update would still trigger a rolling release)")
    call("PATCH", f"/endpoints/{a.endpoint}", body)
    show(a)


def template_update(a):
    body = {}
    if a.image: body["imageName"] = a.image
    if a.disk is not None: body["containerDiskInGb"] = a.disk
    if a.env:                                                   # merge KEY=VAL pairs (empty VAL removes the key)
        env = dict(call("GET", f"/templates/{a.template}").get("env") or {})
        for kv in a.env:
            k, _, v = kv.partition("=")
            if v: env[k] = v
            else: env.pop(k, None)
        body["env"] = env
    if not body:
        sys.exit("nothing to update")
    t = call("PATCH", f"/templates/{a.template}", body)
    print(json.dumps({k: t.get(k) for k in ("id", "name", "imageName", "containerDiskInGb")}, ensure_ascii=False, indent=1))


def billing(a):
    import datetime, urllib.parse
    end = datetime.datetime.now(datetime.timezone.utc); start = end - datetime.timedelta(days=a.days)
    q = urllib.parse.urlencode({"endpointId": a.endpoint, "startTime": start.isoformat(timespec="seconds").replace("+00:00", "Z"),
                                "endTime": end.isoformat(timespec="seconds").replace("+00:00", "Z"), "bucketSize": a.bucket})
    print(json.dumps(call("GET", f"/billing/endpoints?{q}"), ensure_ascii=False, indent=1)[:6000])


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("--image", required=True); c.add_argument("--name", default="h3-video-4090")
    # the image is ~8 GB uncompressed; 30 GB leaves room for /tmp (frames, mp4) — qwen showed a too-small disk kills workers silently
    c.add_argument("--max-workers", type=int, default=1); c.add_argument("--disk", type=int, default=30); c.add_argument("--idle", type=int, default=60)
    c.add_argument("--template", help="reuse an existing template id instead of creating one")
    c.add_argument("--extra-frame-host", action="append", help="additional keyframe host for tests, e.g. raw.githubusercontent.com"); c.set_defaults(fn=create)
    s = sub.add_parser("show"); s.add_argument("--endpoint", required=True); s.set_defaults(fn=show)
    u = sub.add_parser("update"); u.add_argument("--endpoint", required=True); u.add_argument("--max-workers", type=int); u.add_argument("--min-workers", type=int)
    u.add_argument("--idle", type=int); u.add_argument("--timeout-ms", type=int); u.set_defaults(fn=update)
    t = sub.add_parser("template-update"); t.add_argument("--template", required=True); t.add_argument("--image"); t.add_argument("--disk", type=int)
    t.add_argument("--env", nargs="*", help="KEY=VAL to merge (KEY= removes)"); t.set_defaults(fn=template_update)
    b = sub.add_parser("billing"); b.add_argument("--endpoint", required=True); b.add_argument("--days", type=int, default=1)
    b.add_argument("--bucket", default="hour", choices=["hour", "day", "week", "month"]); b.set_defaults(fn=billing)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
