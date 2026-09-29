"""Keyframe download (allow-listed hosts only) and video upload to Cloudflare R2 (S3 API)."""
import io, os, re
from urllib.parse import urlparse

import requests
from PIL import Image

MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 50_000_000           # decompression-bomb guard
MAGIC = {b"\xff\xd8\xff": "jpeg", b"\x89PNG\r\n\x1a\n": "png", b"RIFF": "webp"}
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class BadInput(Exception):
    pass


def allowed_hosts():
    """FRAME_URL_ALLOW: comma-separated host names; an entry starting with '.' also matches its subdomains."""
    return [h.strip().lower() for h in os.environ.get("FRAME_URL_ALLOW", "").split(",") if h.strip()]


def host_allowed(host, allow):
    host = (host or "").lower()
    return any(host == a or (a.startswith(".") and host.endswith(a)) for a in allow)


def fetch_image(field, url, timeout=30):
    if not isinstance(url, str) or not url:
        raise BadInput(f"{field} 必须是 URL 字符串")
    u = urlparse(url)
    if u.scheme != "https":
        raise BadInput(f"{field} 必须是 https URL")
    if not host_allowed(u.hostname, allowed_hosts()):
        raise BadInput(f"{field} 的域名 {u.hostname} 不在白名单内")
    try:
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=False) as r:
            if r.status_code != 200:
                raise BadInput(f"{field} 下载失败 HTTP {r.status_code}")
            buf = io.BytesIO()
            for chunk in r.iter_content(1 << 20):
                buf.write(chunk)
                if buf.tell() > MAX_FRAME_BYTES:
                    raise BadInput(f"{field} 超过 {MAX_FRAME_BYTES >> 20} MiB 上限")
    except requests.RequestException as e:
        raise BadInput(f"{field} 下载失败: {type(e).__name__}")
    return decode_image(field, buf.getvalue())


def decode_image(field, blob):
    if not blob:
        raise BadInput(f"{field} 为空")
    fmt = next((v for m, v in MAGIC.items() if blob.startswith(m)), None)
    if fmt == "webp" and blob[8:12] != b"WEBP":
        fmt = None
    if not fmt:
        raise BadInput(f"{field} 不是 JPEG / PNG / WebP")
    try:
        im = Image.open(io.BytesIO(blob))          # header only; the pixel guard runs before decoding
        if im.width * im.height > MAX_PIXELS:
            raise BadInput(f"{field} 像素数 {im.width}x{im.height} 超过上限")
        im.load()                                   # first frame only for animated WebP/APNG
    except BadInput:
        raise
    except Exception as e:
        raise BadInput(f"{field} 无法解码: {type(e).__name__}")
    return im


def video_key(job_id):
    if not JOB_ID_RE.match(job_id or ""):
        raise ValueError(f"unexpected job id {job_id!r}")
    return f"{os.environ.get('R2_PREFIX', 'videos').strip('/')}/{job_id}.mp4"


_client = None


def r2():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config
        _client = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
                               aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                               config=Config(retries={"max_attempts": 5, "mode": "standard"}, connect_timeout=10, read_timeout=120,
                                             request_checksum_calculation="when_required", response_checksum_validation="when_required"))
    return _client


def upload_video(path, key):
    r2().upload_file(path, os.environ["R2_BUCKET"], key, ExtraArgs={"ContentType": "video/mp4"})
    return os.path.getsize(path)
