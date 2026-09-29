# h3-video-worker 部署方案（草案 v1，2026-09-29）

把 AutoDL 上跑通的 MiniMax-H3（DaSiWa Hybrid 8-step）做成 **RunPod Serverless** endpoint，
由现有的 **Cloudflare Worker** 调用，对外公开提供视频生成服务。
工程结构照搬 `~/qwen-image-edit-4090`（已验证的 4090 / CUDA 13 / ComfyUI worker）。

## 1. 已定决策

| 项 | 决定 |
|---|---|
| 服务对象 | 对外公开、商用（许可证已具备） |
| 网关 | 现有 Cloudflare Worker：鉴权、配额、限流、输入审核、任务记录、接 webhook |
| Prompt 改写 | 在 GPU worker 内，OpenH3-IR 0.4.1 + Qwen3-VL-235B（OpenRouter），与模型加载**并行**；FL2V（首+尾帧）改走 Qwen + `h3-prompt-writing` skill；`skip_rewrite` 直通 |
| 输出存储 | Cloudflare R2（worker 用 S3 API 上传），key = `videos/{job_id}.mp4`，由 worker 生成 |
| 关键帧输入 | R2 预签名 URL（不传 base64） |
| 权重 | RunPod **Network Volume**（镜像只含环境） |
| GPU | **RTX 4090**，1 卡/worker |
| 机房 | **尽量用美国机房**；Volume 首选 US-CA-2 |
| 伸缩 | workersMin 0，workersMax 先 3（账户配额剩 7/30） |
| 仓库 | 独立 GitHub 仓库 `rockycqu002/h3-video-worker`，Actions 构建 → GHCR |

## 2. 架构

```
用户 ──► CF Worker ──(1) 鉴权/配额/审核, 关键帧存 R2
             │
             └─(2) POST api.runpod.ai/v2/<EP>/run  {input, webhook}
                         │
                 RunPod GPU worker (4090, /runpod-volume = 权重)
                   ├─ 启动: ComfyUI :8188 + h3ir :8420（后台常驻）
                   ├─ 下载关键帧 → cover-crop
                   ├─ 改写 ‖ 等 ComfyUI 就绪        (并行)
                   ├─ 构建 workflow → 提交 → WS 等结果 (progress_update)
                   └─ 上传 R2 videos/{job_id}.mp4
                         │
             ◄─(3) webhook: COMPLETED/FAILED + output
CF Worker 更新任务 → 返回 R2 链接给用户
```

## 3. Handler 接口

### 输入 `input`
| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `request` | string | 必填 | 用户原始描述；`skip_rewrite=true` 时视为最终 H3 prompt |
| `skip_rewrite` | bool | false | |
| `creativity` | `restrained｜balanced｜bold｜extreme` | balanced | 仅改写时用（open-h3-ir 的取值） |
| `seconds` | number 4–15 | 5 | 帧数 = max(5, round(s·24)) 上取到 17k+5 |
| `aspect` | `auto｜21:9｜16:9｜4:3｜1:1｜3:4｜9:16` | auto | auto：有首帧跟首帧比例，否则 16:9 |
| `quality` | `768｜640｜544｜416`（短边） | 768 | 长边上限 1344 |
| `seed` | int | 随机 | 结果回传实际 seed |
| `steps` | int 6–12 | 8 | |
| `request` 长度 | ≤ 6000 字符 | | |
| `first_frame_url` / `last_frame_url` | https URL | null | 仅允许 R2 域名（白名单，防 SSRF） |

### 输出（成功）
```json
{
  "video_key": "videos/<job_id>.mp4",
  "width": 1344, "height": 768, "frames": 124, "fps": 24, "seed": 12345, "steps": 8,
  "prompt": "<送进模型的最终 prompt>",
  "rewrite": {"engine": "openh3ir|skill|none", "status": "ok", "fallback_reason": null, "warnings": [], "tokens": 812, "seconds": 11.2},
  "timing": {"boot": 95.0, "rewrite": 11.2, "generate": 190.3, "upload": 2.1},
  "build": "v0.1.0"
}
```

### 输出（失败）
RunPod 状态为 FAILED，`error` 为字符串 `"<code>: <message>"`（CF Worker 按第一个 `: ` 切分），code 取值：
`bad_input`（参数/图片不合法）· `rewrite_failed` · `comfy_rejected` · `oom` · `generate_timeout` · `upload_failed` · `internal`。
CF Worker 按 code 决定是否退额度（`bad_input` 不退，其余退）。

## 4. 仓库结构

```
h3-video-worker/
  Dockerfile                  # cuda 13 base(digest) → torch 2.14+cu130 → ComfyUI 0.37.0(88ab4a0) → open-h3-ir(fd031e13, 独立 venv)，无权重
  constraints.txt
  requirements.txt            # runpod, boto3, aiohttp, pillow ...
  comfy/extra_model_paths.yaml  # → /runpod-volume/h3/models
  workflows/fl2va_api.json
  src/
    handler.py                # runpod 入口、并行编排、错误码
    workflow.py               # 从 h3_deploy/app/workflow.py 移植（去掉官方 ckpt/LoRA 分支）
    rewrite.py                # openh3ir 客户端 + skill 回退
    skill/                    # h3-prompt-writing SKILL.md + base-en.txt
    r2.py                     # 上传/下载
  scripts/
    start.sh                  # 拉起 ComfyUI + h3ir，然后 exec handler
    fetch_models.py           # 在临时 Pod 上把权重拉进 Volume 并校验 sha256
  deploy/runpod_api.py        # 仿 qwen：create/show/update/billing，拒绝改其它 endpoint
  tests/                      # handler 单测 + rp.py 在线协议/压测
  .github/workflows/build.yml # 照搬 qwen（镜像小，不需要 maximize-build-space 那么激进）
```

## 5. 权重（Network Volume）

- 内容（约 40 GB）：`DasiwaMinimaxH3_dasiwaHybrid8turboV1.safetensors` 21 G，Comfy-Org 文本编码器 `nvfp4_awq`，视频 VAE，音频 VAE。
- Volume 大小 60 GB（留余量），位置见 §8 待定。
- 灌装：在同机房开一台最便宜的 CPU/GPU Pod 挂 Volume（Pod 上挂载点是 `/workspace`，serverless 上是 `/runpod-volume`，目录用 `h3/models`），跑 `fetch_models.py` 从 HuggingFace 下载 + sha256 校验，写 `MANIFEST.json`，关 Pod。
- 代价：endpoint 只能调度到该机房的 4090。

## 6. 运行参数

- ComfyUI：`--disable-nvml-pressure`；cu130 下验证能否去掉 `--enable-triton-backend`（comfy_kitchen CUDA 后端）；`--reserve-vram` 视实测。
- Endpoint：queue-based · 4090 · minCuda 13.0 · workersMin 0 · workersMax 3 · idleTimeout 30–60 s（视频任务长，给连续请求留热机）· executionTimeout 900 s · FlashBoot 开 · container disk 30 GB。
- Secrets（endpoint env）：`OPENROUTER_API_KEY`、`OPENROUTER_MODEL=qwen/qwen3-vl-235b-a22b-instruct`、`R2_ENDPOINT`、`R2_BUCKET`、`R2_ACCESS_KEY_ID`、`R2_SECRET_ACCESS_KEY`（仅该 bucket 写权限）、`FRAME_URL_ALLOW`（R2 域名白名单）。

## 7. 实施步骤（每步完成后再确认下一步）

1. 建仓库骨架，移植 workflow/rewrite，写 handler + 单测（本地 CPU 可跑的部分）。
2. 推 GitHub，Actions 构建镜像到 GHCR。
3. 建 Network Volume + 临时 Pod 灌权重。
4. 用临时 **GPU Pod**（4090，挂 Volume，跑同一镜像）手动冒烟：T2V / I2V / FL2V 各一条，量内存峰值、显存、耗时。
5. 建 endpoint（max 1），curl 走 `/run` + `/status`：测冷启动、热启动、webhook、R2 上传、错误码。
6. 压测 + 账单核算（每条 5 s 视频成本），调 idleTimeout / workersMax。
7. 交接给 CF Worker：写 `docs/API_CALLER.md`（仿 qwen），CF 侧加审核、R2 预签名、webhook 接收。

## 8. 待定 / 风险

- **Volume 机房（限美国）**：支持 Volume 的美国机房有 US-CA-2、US-CO-1、US-IL-1、US-MO-2、US-NC-2、US-NE-1、US-TX-3。serverless 各机房 4090 余量无法通过 API 准确查询；qwen worker 多落在 US-CA-2、US-TX-3，故首选 **US-CA-2**。需确认 RunPod 当前是否支持一个 endpoint 挂多个机房的 Volume，若支持再加 US-TX-3 做冗余。
- **系统内存**：AutoDL 上 cgroup 上限 120 GB 内跑通，4090 serverless 主机内存需在第 4 步实测峰值。
- **24 GB 顶满**：AutoDL 峰值显存 23.8 GB，4090 无余量；若 serverless 主机上 OOM，退路是降默认 quality 或改用 5090。
- **预热**：不做 qwen 那样的开机试跑（一次生成要几分钟，太贵）；改为开机时后台把权重顺序读一遍进页缓存，与第一单的改写并行。主机内存若小于权重体积则收益有限，第 5 步用 `H3_PREFETCH=0/1` 对比。
- **冷启动**：Volume 读 40 GB + 模型加载，预计 1–3 分钟，第 5 步实测；FlashBoot 能否缓解待测。
- **配额**：账户 max worker 23/30，H3 先占 3；上量前需向 RunPod 申请提额。
- **成本**：4090 Pod 价 $0.74/h，serverless 更高；按 ~200 s/条粗估每条几美分量级，第 6 步用 billing 接口核实。
- **OpenRouter 余额**：公开服务后需监控。
