# H3 视频生成 endpoint 调用说明（给 Cloudflare Worker）

调用方是 Cloudflare Worker：用户请求 → Worker 做鉴权、配额、**输入审核** → 把关键帧存进 R2 并生成预签名 URL →
`POST /run`（带 webhook）→ RunPod worker 生成视频并写入 R2 → webhook 通知 Worker → Worker 把视频交给用户。

RunPod API key、R2 凭据只能放在 Worker 的 secret 里，不能下发到浏览器。协议以 `src/handler.py` 为准。

---

## 1. 基本信息

| 项 | 值 |
|---|---|
| Endpoint ID | 测试：`ekb0yn2gdsmyjo`（`h3-video-4090-test`）；正式 endpoint 上线后替换，**用环境变量 `RUNPOD_H3_ENDPOINT_ID`，不要写死** |
| API 根地址 | `https://api.runpod.ai/v2/<ENDPOINT_ID>` |
| 鉴权 | `Authorization: Bearer <RUNPOD_API_KEY>` |
| 调用模式 | 队列式：`POST /run` 提交（带 `webhook`）；`GET /status/{id}` 兜底查询。**不要用 `/runsync`**（任务要几分钟） |
| GPU / 模型 | RTX 4090 24 GB · MiniMax H3 DaSiWa Hybrid 8 步 · 24 fps · 视频 + 立体声音频 |
| 机房 | US-CA-2（权重所在 Network Volume 的机房） |
| 单条耗时（5 s @ 1344×768，worker 已热） | 生成 170–200 s + 改写 7–30 s ≈ **3–4 分钟** |
| 冷启动（没有热 worker） | 额外排队 26–160 s（2026-09-29 实测：首次 162 s，之后 26 s） |
| 时长与耗时 | 生成时间大致随帧数线性增长：10 s ≈ 2×，15 s ≈ 3×（未实测，按帧数估） |
| 执行超时 | 900 s（worker 在 840 s 主动中止并返回 `generate_timeout`） |
| 结果保留 | `/run` 的结果在 RunPod 侧保留约 **30 分钟**；视频本身在 R2，不受影响 |
| 扩缩容 | 0–N 个 worker（测试 endpoint N=1），每个 worker 同时只跑 1 个任务，多余请求排队 |
| 成本（估算） | 4090 worker $0.74/h，热启动一条 5 s 视频 ≈ $0.04；冷启动多 $0.01–0.03；改写（OpenRouter）< $0.01。以 RunPod 账单为准 |

---

## 2. 提交任务 `POST /run`

```http
POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run
Authorization: Bearer <RUNPOD_API_KEY>
Content-Type: application/json

{
  "input": {
    "request": "一位咖啡师在洒满阳光的咖啡馆里拉花，然后抬头微笑",
    "seconds": 5,
    "first_frame_url": "https://21c38efd61d6ef0a93541eb0cb1bc3fa.r2.cloudflarestorage.com/h3-videos/frames/<task>/first.jpg?X-Amz-...",
    "seed": 12345
  },
  "webhook": "https://<你的 Worker 域名>/runpod/h3-webhook?t=<task_id>&sig=<HMAC>"
}
```

| 字段 | 必填 | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| `input.request` | 是 | string | — | 用户的自然语言需求，中英文均可，≤ 6000 字符。`skip_rewrite=true` 时视为**最终 H3 prompt** 原样送入模型 |
| `input.skip_rewrite` | 否 | bool | `false` | 跳过自动改写。仅给懂 H3 prompt 格式的高级用户 / 内部调试 |
| `input.creativity` | 否 | string | `balanced` | 改写的发挥程度：`restrained` / `balanced` / `bold` / `extreme`。`bold` 以上会自行添加画面元素 |
| `input.seconds` | 否 | number | `5` | 4–15。实际帧数 = 对齐到 H3 的 17k+5 网格（5 s → 124 帧，15 s → 362 帧） |
| `input.aspect` | 否 | string | `auto` | `auto` / `21:9` / `16:9` / `4:3` / `1:1` / `3:4` / `9:16`。`auto`：有关键帧则跟随关键帧比例，否则 16:9 |
| `input.quality` | 否 | string | `768` | 短边像素：`768` / `640` / `544` / `416`（长边上限 1344）。越小越快 |
| `input.seed` | 否 | int | 随机 | 0–2^53；实际值在 `output.seed` 返回 |
| `input.steps` | 否 | int | `8` | 6–12。DaSiWa 为 8 步蒸馏模型，**保持默认** |
| `input.first_frame_url` | 否 | string | — | 首帧图片的 **https** URL（见 §6） |
| `input.last_frame_url` | 否 | string | — | 尾帧图片的 **https** URL |
| `webhook` | 建议 | string | — | 任务结束（成功或失败）时 RunPod 向该地址 POST 结果（见 §5） |

模式由关键帧自动决定：无图 = T2V；只有首帧 = I2V；只有尾帧 = L2V；首尾都有 = FL2V。
未知字段被忽略。响应（HTTP 200）：`{"id": "6e9a7b15-…-u1", "status": "IN_QUEUE"}` —— **这个 `id` 就是 job_id，务必落库**，
视频 key 由它决定（`videos/<job_id>.mp4`）。

---

## 3. 查询状态 `GET /status/{id}`（兜底用）

`status`：`IN_QUEUE` → `IN_PROGRESS` → `COMPLETED` | `FAILED` | `CANCELLED` | `TIMED_OUT`。
`delayTime`（排队 + 冷启动，ms）和 `executionTime`（worker 内执行，ms）由 RunPod 填写。

进行中时 `output` 是一个**进度字符串**，可直接映射成 UI 文案：

| `output` | 含义 |
|---|---|
| `preparing` | 下载 / 裁剪关键帧 |
| `rewriting` | 正在改写 prompt（同时等待冷启动） |
| `generating` | 已提交到模型 |
| `generating 45s` | 已生成 45 s（每 15 s 更新一次；5 s 视频总共约 170–200 s） |
| `uploading` | 写入 R2 |

成功：

```json
{
  "id": "6e9a7b15-1d70-480e-907c-d88dcd5f1df4-u1",
  "status": "COMPLETED",
  "delayTime": 26036,
  "executionTime": 183512,
  "output": {
    "video_key": "videos/6e9a7b15-1d70-480e-907c-d88dcd5f1df4-u1.mp4",
    "bytes": 1868924,
    "width": 1344, "height": 768, "frames": 124, "fps": 24, "seconds": 5,
    "seed": 42, "steps": 8, "mode": "t2v",
    "prompt": "integrated_multimodal_description: [Shot 1] Live-action, cinematic, …",
    "rewrite": {"engine": "openh3ir", "status": "ready", "fallback_reason": null, "warnings": ["…"], "tokens": 284, "seconds": 13.6},
    "timing": {"boot": 5, "rewrite": 13.6, "generate": 169.1, "upload": 0.5, "total": 183.2},
    "build": "v0.1.0",
    "worker": {"jobs": 1, "comfy_ready_s": 8.6, "prefetch_s": 41.2, "vram_used_mib": 22326, "mem_mib": {"current": "…", "peak": "…", "max": "…", "anon": "…"}}
  }
}
```

| 字段 | 用途 |
|---|---|
| `video_key` | R2 bucket `h3-videos` 里的对象 key。mp4（H.264 视频 + 32 kHz 立体声 AAC） |
| `prompt` | 实际送进模型的最终 prompt。建议落库（复现、排查、二次编辑） |
| `rewrite.engine` | `openh3ir`（T2V/I2V/L2V 默认）/ `skill`（FL2V，或 openh3ir 失败时的回退）/ `none`（skip_rewrite） |
| `seed` | 同 seed + 同 prompt + 同关键帧可复现 |
| `timing` / `worker` | 监控用，不影响业务逻辑 |

`worker.mem_mib` 自 v0.1.1 起提供（v0.1.0 为 `ram_used_mib`，是宿主机数值，忽略即可）。

---

## 4. 失败语义

失败时 `status = "FAILED"`，**错误在顶层 `error` 字段**（字符串，没有 `output`），格式固定为 `"<code>: <message>"`：

```json
{"id": "ea23cd6e-…-u2", "status": "FAILED", "delayTime": 7792, "executionTime": 115,
 "error": "bad_input: seconds=99.0 超出允许范围 4.0–15.0"}
```

按第一个 `": "` 切分得到 `code`：

| code | 含义 | 重试 | 退额度 |
|---|---|---|---|
| `bad_input` | 参数越界、关键帧 URL 不合法 / 不在白名单 / 下载失败 / 不是 JPEG·PNG·WebP / > 16 MiB / > 50 MP | 否（message 可直接展示，中文） | 否（前置校验应拦住） |
| `rewrite_failed` | 改写失败（OpenRouter 不可用 / 额度不足 / 模型输出不合格，已重试 3 次） | 可重试 1 次，或提示用户稍后再试 | 是 |
| `comfy_rejected` | 模型执行报错 | 可重试 1 次 | 是 |
| `oom` | 显存不足（worker 会自动被替换） | 可换较低 `quality` / 较短 `seconds` 重试 | 是 |
| `generate_timeout` | 超过 840 s | 可降 `seconds` 重试 | 是 |
| `upload_failed` | 写 R2 失败（凭据 / 网络） | 可重试 1 次；持续出现说明 R2 token 失效 | 是 |
| `internal` | 其他（含权重缺失、ComfyUI 崩溃） | 重试 1 次 | 是 |
| （无 code 前缀） | 平台级错误，如 `executionTimeout exceeded`；或 `status` 为 `TIMED_OUT` / `CANCELLED` | 重试 1 次 | 是 |

对已发出但没拿到响应的 `/run` 不要盲目重试——平台可能已接收并计费；先用自己记录的 task 查一下是否拿到过 job id。

---

## 5. Webhook

`/run` 请求体顶层带 `"webhook": "<url>"` 时，任务进入终态后 RunPod 会向该 URL `POST` 与 `/status` **相同结构**的 JSON。

- **RunPod 不对 webhook 签名。** 在 URL 里带上自己的校验参数（如 `?t=<task_id>&sig=HMAC(secret, task_id)`），
  收到后校验签名，并确认 body 里的 `id` 等于该 task 记录的 job id。
- 尽快返回 HTTP 200（写库后即返回）；RunPod 对非 200 只做有限次重试，**不能保证必达**。
- 处理要幂等：同一个 job 可能收到多次。
- **兜底**：用 Cron Trigger 每分钟扫一次“已提交超过 10 分钟仍未终态”的任务，对其调 `/status/{id}`
  （RunPod 只保留结果约 30 分钟，务必在这之前补拉）。

---

## 6. 关键帧：R2 预签名 URL

worker 只从白名单域名下载关键帧，当前白名单：`21c38efd61d6ef0a93541eb0cb1bc3fa.r2.cloudflarestorage.com`
（R2 的 S3 API 域名）。**换用自定义域名前先通知运维把它加入 `FRAME_URL_ALLOW`**，否则会返回 `bad_input`。

流程：用户上传图片 → Worker 用 R2 binding 写入 `h3-videos/frames/<task_id>/first.jpg` → 用 S3 凭据生成 **GET 预签名 URL** →
放进 `first_frame_url`。

- 有效期至少 **2 小时**：worker 在任务开始执行时才下载，排队 + 冷启动可能要几分钟。
- 格式 JPEG / PNG / WebP，≤ 16 MiB，≤ 50 MP；EXIF 方向会被自动应用，透明区域合成白底。
- worker 会把关键帧**居中裁剪**到画布比例（不拉伸）。`aspect=auto` 时画布跟随首帧（没有首帧则跟随尾帧）。
- 预签名需要 S3 凭据（R2 binding 本身不能签名）。建议为 Worker 单独建一个 **h3-videos 只读** 的 R2 token
  （签 GET 只需要读权限；写入用 binding），不要复用 RunPod 那把读写 token。

## 7. 交付视频

bucket `h3-videos` 不公开。两种方式：

1. **Worker 代理**：`env.H3_VIDEOS.get(video_key)`，把 body 流式返回给用户，自己做鉴权（推荐，可控、可计量）。
2. **预签名 GET**：给用户一个短期（如 1 小时）的下载链接。

视频和关键帧建议配 R2 生命周期规则按天数自动删除（天数待定）。

---

## 8. 输入审核（公开服务必须）

提交到 RunPod **之前**在 Worker 里完成：

- `request` 文本：暴力、色情（尤其涉及未成年人）、仇恨、侵权等。
- 关键帧图片：**真人照片须特别处理**——禁止对真实可识别人物生成色情、侮辱或冒充内容。
- 被拒的请求直接在 Worker 返回，不提交、不计费。

worker 端不做内容审核；模型（DaSiWa，未改动官方安全行为）与改写器也不能替代它。

---

## 9. 示例（Cloudflare Worker，TypeScript）

```ts
import { AwsClient } from "aws4fetch";

interface Env {
  RUNPOD_API_KEY: string;            // secret
  RUNPOD_H3_ENDPOINT_ID: string;     // var
  R2_READ_ACCESS_KEY_ID: string;     // secret: h3-videos 只读 token
  R2_READ_SECRET_ACCESS_KEY: string; // secret
  WEBHOOK_SECRET: string;            // secret: 用于 webhook URL 签名
  H3_VIDEOS: R2Bucket;               // binding → bucket h3-videos
}

const R2_HOST = "21c38efd61d6ef0a93541eb0cb1bc3fa.r2.cloudflarestorage.com";

async function presignGet(env: Env, key: string, seconds = 7200) {
  const r2 = new AwsClient({ accessKeyId: env.R2_READ_ACCESS_KEY_ID, secretAccessKey: env.R2_READ_SECRET_ACCESS_KEY });
  const url = new URL(`https://${R2_HOST}/h3-videos/${key}`);
  url.searchParams.set("X-Amz-Expires", String(seconds));
  return (await r2.sign(new Request(url, { method: "GET" }), { aws: { signQuery: true } })).url;
}

async function hmac(secret: string, msg: string) {
  const k = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", k, new TextEncoder().encode(msg));
  return [...new Uint8Array(sig)].map(b => b.toString(16).padStart(2, "0")).join("");
}

// 提交（审核、配额检查已通过之后）
async function submit(env: Env, origin: string, taskId: string, request: string, seconds: number, first?: ArrayBuffer) {
  const input: Record<string, unknown> = { request, seconds };
  if (first) {
    const key = `frames/${taskId}/first.jpg`;
    await env.H3_VIDEOS.put(key, first, { httpMetadata: { contentType: "image/jpeg" } });
    input.first_frame_url = await presignGet(env, key);
  }
  const webhook = `${origin}/runpod/h3-webhook?t=${taskId}&sig=${await hmac(env.WEBHOOK_SECRET, taskId)}`;
  const r = await fetch(`https://api.runpod.ai/v2/${env.RUNPOD_H3_ENDPOINT_ID}/run`, {
    method: "POST",
    headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({ input, webhook }),
  });
  if (!r.ok) throw new Error(`runpod /run ${r.status}`);
  const { id } = await r.json() as { id: string };
  return id;                                   // 落库：task_id ↔ job_id
}

type H3Status = {
  id: string; status: "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED" | "CANCELLED" | "TIMED_OUT";
  error?: string; delayTime?: number; executionTime?: number;
  output?: string | { video_key: string; prompt: string; seed: number; width: number; height: number; seconds: number };
};

// webhook 接收
async function onWebhook(req: Request, env: Env) {
  const u = new URL(req.url);
  const taskId = u.searchParams.get("t") ?? "";
  if (u.searchParams.get("sig") !== await hmac(env.WEBHOOK_SECRET, taskId)) return new Response("forbidden", { status: 403 });
  const st = await req.json() as H3Status;
  // 1. 查库确认 st.id 就是该 task 的 job_id；已是终态则直接返回 200（幂等）
  if (st.status === "COMPLETED" && typeof st.output === "object") {
    // 2. 记录 st.output.video_key / prompt / seed，标记成功
  } else if (st.status !== "IN_QUEUE" && st.status !== "IN_PROGRESS") {
    const code = st.error?.split(": ", 1)[0] ?? st.status;       // bad_input / rewrite_failed / …
    // 3. 标记失败；code !== "bad_input" 时退额度
  }
  return new Response("ok");
}
```

---

## 10. 验收用例

| # | 用例 | 预期 |
|---|---|---|
| 1 | T2V，5 s，只传 `request` | `COMPLETED`，`mode=t2v`，`rewrite.engine=openh3ir`，1344×768，124 帧 |
| 2 | I2V，竖图首帧（预签名 URL） | `COMPLETED`，`mode=i2v`，768×1344（跟随首帧比例） |
| 3 | FL2V，首尾帧 | `COMPLETED`，`mode=fl2v`，`rewrite.engine=skill`，`prompt` 首行为对齐说明 |
| 4 | `skip_rewrite=true` + 现成 H3 prompt | `COMPLETED`，`rewrite.engine=none`，`prompt` 与输入一致 |
| 5 | `seconds=99` | `FAILED`，`error` 以 `bad_input:` 开头，不退额度、不重试 |
| 6 | `first_frame_url` 用非白名单域名 | `FAILED`，`bad_input: … 不在白名单内` |
| 7 | 预签名 URL 已过期 | `FAILED`，`bad_input: first_frame_url 下载失败 HTTP 403` |
| 8 | 无热 worker 时提交 | `IN_QUEUE` 停留数十秒到约 3 分钟后开始，webhook 正常到达 |
| 9 | webhook 签名错误 / 重复投递 | 403 / 幂等不重复入账 |
| 10 | 故意不处理 webhook | Cron 兜底在 30 分钟内用 `/status` 补拉成功 |

---

## 11. 参考

- 方案与决策：`docs/PLAN.md`
- Worker 源码（协议以此为准）：`src/handler.py`、`src/storage.py`
- 部署 / 更新 endpoint：`deploy/runpod_api.py`
- 在线冒烟测试：`tests/smoke.py`
