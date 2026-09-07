# 天数智芯 智铠100 文本生成引擎（基于 vLLM 优化）

本项目是为**天数智芯-智铠100**加速卡深度优化的高性能文本生成推理引擎，基于开源 **vLLM** 框架进行架构级适配与增强，实现对 **Qwen3 系列**等最新大模型的高效支持。通过引入 **Prefix Caching**、PagedAttention 等先进优化技术，显著提升吞吐与响应速度，同时提供标准 **OpenAI 兼容 API 接口**，便于无缝集成现有应用生态。

## 支持模型

- **Qwen3**
- **Llama3**
- **DeepSeek-R1-Distill**
- 其他兼容 vLLM 的 HuggingFace 模型（持续扩展中）

> 模型下载地址：[https://modelscope.cn/models/Qwen](https://modelscope.cn/models/Qwen)

---

## Quick Start

### 1. 模型下载

从 ModelScope 下载所需模型（以 Qwen2.5-7B-Instruct 为例）：

```bash
modelscope download --model qwen/Qwen2.5-7B-Instruct README.md --local_dir /mnt/models/Qwen2.5-7B-Instruct
```

> ⚠️ 请确保模型路径在后续 Docker 启动时正确挂载。

---

### 2. 拉取并构建 Docker 镜像

我们提供已预装智铠100驱动与vLLM优化版本的Docker镜像：

```
# 本地构建
docker build -t enginex-iluvatar-vllm:bi100 -f Dockerfile .
```

---

### 3. 启动服务容器

```bash
docker run -it --rm -p 8000:80 \
  --name vllm-iluvatar \
  -v /mnt/models/Qwen2.5-7B-Instruct:/model:ro \
  --privileged \
  -e TENSOR_PARALLEL_SIZE=1 \
  -e PREFIX_CACHING=true \
  -e MAX_MODEL_LEN=10000 \
  enginex-iluvatar-vllm:bi100
```

> ✅ 参数说明：
> - `PREFIX_CACHING=true`: 启用 Prefix Caching 优化，显著提升多请求共享前缀的推理效率
> - `MAX_MODEL_LEN=10000`: 支持长上下文推理
> - `--privileged`: 确保智铠100设备可见

---

## 4. 测试服务（使用 OpenAI 兼容接口）

服务启动后，可通过标准 OpenAI SDK 或 `curl` 进行测试。

### 示例：文本生成请求

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "请用中文介绍一下上海的特点。"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### 使用 OpenAI Python SDK（需安装 `openai>=1.0`）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="qwen3-8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请简要介绍杭州的特色文化。"}
    ],
    max_tokens=512,
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

## 测试结果对比（A100 vs 智铠100）

### 测试数据集

[chat_dataset_v0.json](chat_dataset_v0.json)

### 测试结果

在相同模型和输入条件下，测试平均输出速度（单位：字每秒），结果如下：

| 模型名称 | A100出字速度(字/秒) | 出字速度(字/秒) | A100输出质量 | 输出质量 | A100首字延迟(秒) | 首字延迟(秒) | 备注 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| AI-ModelScope/Mistral-7B-Instruct-v0.2 | 50.0638 | 55.3545 | 88.5000 | 85.0000 | 0.2544 | 0.2078 | |
| codefuse-ai/CodeFuse-QWen-14B | 75.4755 | 61.5555 | 63.7500 | 42.5000 | 0.1983 | 0.0840 | |
| Qwen/CodeQwen1.5-7B-Chat | 108.0509 | 98.9290 | 35.0000 | 23.7500 | 0.1042 | 0.1735 | |
| Qwen/Qwen-1_8B-Chat | 203.9495 | 151.5506 | 65.0000 | 60.0000 | 0.0760 | 0.0951 | |
| Qwen/Qwen-7B | 112.5454 | 90.3540 | 55.0000 | 63.7500 | 0.1047 | 0.1419 | |
| Qwen/Qwen-7B-Chat | 131.4390 | 93.7376 | 85.0000 | 65.0000 | 0.1198 | 0.0920 | |
| Qwen/Qwen1.5-1.8B | 104.2267 | 111.8826 | 26.2500 | 37.5000 | 0.1176 | 0.0990 | |
| Qwen/Qwen1.5-14B-Chat | 72.2311 | 52.0573 | 89.7500 | 88.5000 | 0.1024 | 0.1990 | |
| Qwen/Qwen1.5-4B-Chat | 95.4927 | 87.0352 | 85.0000 | 85.0000 | 0.0991 | 0.1340 | |
| Qwen/Qwen2-0.5B-Instruct | 146.2331 | 141.0503 | 50.0000 | 56.2500 | 0.0981 | 0.1003 | |
| Qwen/Qwen2-1.5B | 175.4972 | 161.2327 | 38.7500 | 53.7500 | 0.1036 | 0.1298 | |
| Qwen/Qwen2-1.5B-Instruct | 124.5098 | 119.4177 | 75.0000 | 80.0000 | 0.0895 | 0.1117 | |
| Qwen/Qwen2-7B | 169.9027 | 74.3288 | 72.5000 | 56.2500 | 0.1120 | 0.2810 | |
| Qwen/Qwen2-7B-Instruct | 110.1237 | 59.1989 | 89.2500 | 89.7500 | 0.0971 | 0.2078 | |
| Qwen/Qwen2.5-1.5B-Instruct | 116.6704 | 123.0560 | 85.0000 | 86.7500 | 0.1291 | 0.1216 | |
| Qwen/Qwen2.5-32B-Instruct-AWQ | 47.4427 | 52.5942 | 91.0000 | 87.5000 | 0.1332 | 0.3550 | |
| Qwen/Qwen2.5-3B-Instruct | 95.6249 | 91.6548 | 85.0000 | 85.0000 | 0.1122 | 0.1332 | |
| Qwen/Qwen2.5-72B-Instruct-AWQ | 41.4106 | 30.7387 | 91.0000 | 91.0000 | 0.1366 | 0.7175 | |
| Qwen/Qwen2.5-VL-7B-Instruct-AWQ | 61.5433 | 56.6830 | 88.5000 | 88.5000 | 0.2346 | 0.2472 | |
| Qwen/Qwen3-14B-AWQ | 62.6335 | 53.3993 | 86.7500 | 88.5000 | 0.1429 | 0.2347 | |
| Qwen/Qwen3-32B-AWQ | 44.0649 | 31.8064 | 89.2500 | 88.5000 | 0.2027 | 0.5392 | |
| Qwen/QwQ-32B-AWQ | 47.9752 | 50.8763 | 88.5000 | 88.5000 | 0.2201 | 0.4284 | |
| swift/Qwen3-30B-A3B-AWQ | 38.1391 | 29.5163 | 88.0000 | 88.5000 | 0.1619 | 0.2017 | |
| Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ | 51.5753 | 51.8930 | 88.0000 | 88.0000 | 0.2052 | 0.4199 | |
| X-D-Lab/MindChat-Qwen-7B | 123.3664 | 92.9101 | 71.2500 | 70.0000 | 0.1032 | 0.1271 | |
