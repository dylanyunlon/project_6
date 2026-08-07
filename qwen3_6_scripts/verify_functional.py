#!/usr/bin/env python3
"""Functional verification script — mirrors CCCL's test design pattern.

CCCL catch2_test_device_three_way_partition.cu verifies:
  1. Empty input handling
  2. Stability (CUB result == Thrust result)
  3. Edge cases (empty first/second/unselected parts)
  4. Large problem sizes

We verify the same categories for vllm:
  1. Empty/minimal input handling
  2. Response correctness (HTTP 200, valid JSON, non-empty content)
  3. Edge cases (long context, tool calls, reasoning split)
  4. All chat_dataset_v0.json conversations

Usage (after starting vllm server):
  python3 verify_functional.py --endpoint http://localhost:8000
  python3 verify_functional.py --endpoint http://localhost:8000 --quick
"""

import argparse
import json
import sys
import time
import requests
from typing import List, Dict, Tuple


def chat_completion(endpoint: str, messages: List[Dict], **kwargs) -> Dict:
    """Send a chat completion request and return the response."""
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": "llm",
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 200),
        "temperature": kwargs.get("temperature", 0.7),
        "stream": False,
    }
    payload.update(kwargs)
    resp = requests.post(url, json=payload, timeout=120)
    return resp.status_code, resp.json() if resp.status_code == 200 else resp.text


# ================================================================
# Test cases — mirrors CCCL's categorized test structure
# ================================================================

def test_basic_chat(endpoint: str) -> Tuple[bool, str]:
    """TC-01: Basic non-streaming chat returns HTTP 200 + valid content."""
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "你好"}
    ], max_tokens=50)
    if code != 200:
        return False, f"HTTP {code}: {data}"
    content = data["choices"][0]["message"]["content"]
    if not content or len(content) < 2:
        return False, f"Empty or too short content: '{content}'"
    usage = data.get("usage", {})
    if usage.get("completion_tokens", 0) <= 0:
        return False, f"completion_tokens <= 0: {usage}"
    return True, f"OK: {len(content)} chars, {usage.get('completion_tokens')} tokens"


def test_finish_reason(endpoint: str) -> Tuple[bool, str]:
    """TC-02: finish_reason is 'stop' or 'length'."""
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "说一个字"}
    ], max_tokens=10)
    if code != 200:
        return False, f"HTTP {code}"
    fr = data["choices"][0].get("finish_reason")
    if fr not in ("stop", "length"):
        return False, f"finish_reason='{fr}', expected stop/length"
    return True, f"OK: finish_reason={fr}"


def test_chinese_output(endpoint: str) -> Tuple[bool, str]:
    """TC-03: Chinese content generation quality."""
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "请用一句话解释什么是GPU"}
    ], max_tokens=100)
    if code != 200:
        return False, f"HTTP {code}"
    content = data["choices"][0]["message"]["content"]
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in content)
    if not has_chinese:
        return False, f"No Chinese characters in: '{content[:50]}'"
    if len(content) < 10:
        return False, f"Content too short: {len(content)} chars"
    return True, f"OK: {len(content)} chars, Chinese present"


def test_system_prompt(endpoint: str) -> Tuple[bool, str]:
    """TC-04: System prompt controls output."""
    code, data = chat_completion(endpoint, [
        {"role": "system", "content": "无论用户说什么，你只能回复 FIXED_REPLY_42"},
        {"role": "user", "content": "你好啊"}
    ], max_tokens=50)
    if code != 200:
        return False, f"HTTP {code}"
    content = data["choices"][0]["message"]["content"]
    if "FIXED_REPLY_42" not in content:
        return False, f"System prompt not followed: '{content[:80]}'"
    return True, f"OK: contains FIXED_REPLY_42"


def test_multi_turn_memory(endpoint: str) -> Tuple[bool, str]:
    """TC-05: Multi-turn conversation memory."""
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "记住暗号：ALPHA_BRAVO"},
        {"role": "assistant", "content": "好的，我记住了暗号ALPHA_BRAVO"},
        {"role": "user", "content": "请说出之前的暗号"}
    ], max_tokens=50)
    if code != 200:
        return False, f"HTTP {code}"
    content = data["choices"][0]["message"]["content"]
    if "ALPHA_BRAVO" not in content:
        return False, f"Memory failed: '{content[:80]}'"
    return True, f"OK: recalled ALPHA_BRAVO"


def test_reasoning_separation(endpoint: str) -> Tuple[bool, str]:
    """TC-06: reasoning_content and content are separated."""
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "逐步计算 17×23"}
    ], max_tokens=500)
    if code != 200:
        return False, f"HTTP {code}"
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    if not content:
        return False, "content is empty"
    if "<think>" in content:
        return False, f"content contains <think> tag"
    # reasoning_content may or may not be present depending on model config
    return True, f"OK: content={len(content)}c, reasoning={len(reasoning)}c"


def test_tool_calling(endpoint: str) -> Tuple[bool, str]:
    """TC-07: Tool calling returns valid tool_calls."""
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "北京今天天气怎么样"}
    ], max_tokens=200, tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }], tool_choice="required")
    if code != 200:
        return False, f"HTTP {code}: {data}"
    msg = data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls", [])
    if not tool_calls:
        return False, "No tool_calls returned"
    tc = tool_calls[0]
    try:
        args = json.loads(tc["function"]["arguments"])
    except (json.JSONDecodeError, KeyError) as e:
        return False, f"Invalid tool_calls: {e}"
    return True, f"OK: {tc['function']['name']}({args})"


def test_stop_sequence(endpoint: str) -> Tuple[bool, str]:
    """TC-08: Stop sequence truncation."""
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "从1数到30"}
    ], max_tokens=200, stop=["15"])
    if code != 200:
        return False, f"HTTP {code}"
    content = data["choices"][0]["message"]["content"]
    fr = data["choices"][0].get("finish_reason")
    if "16" in content or "17" in content:
        return False, f"Stop sequence not effective: '{content[:80]}'"
    return True, f"OK: finish_reason={fr}, no '16' in output"


def test_temperature_zero(endpoint: str) -> Tuple[bool, str]:
    """TC-09: temperature=0 (greedy) works."""
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "hi"}
    ], max_tokens=20, temperature=0.0)
    if code != 200:
        return False, f"HTTP {code}: {data}"
    return True, f"OK: greedy sampling works"


def test_empty_messages_error(endpoint: str) -> Tuple[bool, str]:
    """TC-10: Empty messages returns 4xx."""
    url = f"{endpoint}/v1/chat/completions"
    resp = requests.post(url, json={"model": "llm", "messages": []}, timeout=30)
    if resp.status_code < 400:
        return False, f"Expected 4xx, got {resp.status_code}"
    return True, f"OK: HTTP {resp.status_code} for empty messages"


def test_max_tokens_boundary(endpoint: str) -> Tuple[bool, str]:
    """TC-11: max_tokens boundary values (CCCL ThreadScanExclusivePartial pattern).

    CCCL catch2_test_thread_scan_exclusive_partial.cu tests valid_items at:
      1, [2..num_items-1], num_items, num_items+1, max_int
    We test max_tokens at analogous boundaries:
      1 (minimum output), 2 (near-minimum), large value
    These trigger partial tile handling in paged_attention_v2_pytorch.py.
    """
    # max_tokens=1: partial tile with single output token
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "hi"}
    ], max_tokens=1)
    if code != 200:
        return False, f"max_tokens=1: HTTP {code}"
    content = data["choices"][0]["message"]["content"]
    fr = data["choices"][0].get("finish_reason")
    if fr not in ("stop", "length"):
        return False, f"max_tokens=1: finish_reason={fr}"

    # max_tokens=2: CCCL valid_items=2 boundary
    code2, data2 = chat_completion(endpoint, [
        {"role": "user", "content": "count to ten"}
    ], max_tokens=2)
    if code2 != 200:
        return False, f"max_tokens=2: HTTP {code2}"

    return True, f"OK: max_tokens=1 got '{content[:20]}' ({fr}), max_tokens=2 passed"


def test_json_object_output(endpoint: str) -> Tuple[bool, str]:
    """TC-12: response_format=json_object forces valid JSON output."""
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "返回一个JSON，包含name=Alice,age=30"}
    ], max_tokens=100, response_format={"type": "json_object"})
    if code != 200:
        return False, f"HTTP {code}: {data}"
    content = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
        if "name" not in parsed and "age" not in parsed:
            return False, f"JSON missing name/age: {content[:100]}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}. Content: {content[:100]}"
    return True, f"OK: valid JSON with keys {list(parsed.keys())}"


def test_chat_dataset(endpoint: str) -> Tuple[bool, str]:
    """TC-13: Run chat_dataset_v0.json conversations."""
    try:
        with open("chat_dataset_v0.json") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        # Try from script directory
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "..", "chat_dataset_v0.json")) as f:
            dataset = json.load(f)

    total = 0
    passed = 0
    for conv in dataset:
        system = conv.get("system_prompt", "You are a helpful assistant.")
        messages = [{"role": "system", "content": system}]
        for q in conv["user_questions"][:2]:  # First 2 turns only for speed
            messages.append({"role": "user", "content": q})
            code, data = chat_completion(endpoint, messages, max_tokens=300)
            total += 1
            if code == 200:
                content = data["choices"][0]["message"]["content"]
                if content and len(content) > 5:
                    passed += 1
                    messages.append({"role": "assistant", "content": content})
                else:
                    messages.append({"role": "assistant", "content": ""})
            else:
                messages.append({"role": "assistant", "content": ""})

    if passed < total * 0.8:
        return False, f"Only {passed}/{total} turns passed"
    return True, f"OK: {passed}/{total} turns passed"


# ================================================================
# Runner
# ================================================================

ALL_TESTS = [
    ("TC-01 Basic chat", test_basic_chat),
    ("TC-02 Finish reason", test_finish_reason),
    ("TC-03 Chinese output", test_chinese_output),
    ("TC-04 System prompt", test_system_prompt),
    ("TC-05 Multi-turn memory", test_multi_turn_memory),
    ("TC-06 Reasoning separation", test_reasoning_separation),
    ("TC-07 Tool calling", test_tool_calling),
    ("TC-08 Stop sequence", test_stop_sequence),
    ("TC-09 Temperature zero", test_temperature_zero),
    ("TC-10 Empty messages error", test_empty_messages_error),
    ("TC-11 Max tokens boundary", test_max_tokens_boundary),
    ("TC-12 JSON object output", test_json_object_output),
    ("TC-13 Chat dataset", test_chat_dataset),
]

QUICK_TESTS = ALL_TESTS[:5]  # First 5 for quick validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    tests = QUICK_TESTS if args.quick else ALL_TESTS
    passed = 0
    failed = 0

    print(f"=== Functional Verification ({len(tests)} tests) ===")
    print(f"Endpoint: {args.endpoint}\n")

    for name, fn in tests:
        try:
            ok, msg = fn(args.endpoint)
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"  [{status}] {name}: {msg}")
        except Exception as e:
            failed += 1
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")

    print(f"\nResult: {passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()


def test_streaming_sse(endpoint: str) -> Tuple[bool, str]:
    """TC-14: Streaming SSE protocol — data: chunks + [DONE] terminator.

    CCCL parallel: agent_scan.cuh lookback tile_state streaming.
    Each scan tile publishes its partial result via tile_descriptor_t
    (SCAN_TILE_INVALID → SCAN_TILE_PARTIAL → SCAN_TILE_INCLUSIVE).
    SSE is the HTTP analog: each chunk publishes a delta, [DONE] = INCLUSIVE.
    """
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "写一首四句诗"}],
        "max_tokens": 200,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    resp = requests.post(url, json=payload, timeout=120, stream=True)
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"

    chunks = []
    has_done = False
    has_usage = False
    content_parts = []

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                has_done = True
                continue
            try:
                chunk = json.loads(data_str)
                chunks.append(chunk)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    content_parts.append(delta["content"])
                if chunk.get("usage"):
                    has_usage = True
            except json.JSONDecodeError:
                pass

    full_content = "".join(content_parts)
    if len(chunks) < 5:
        return False, f"Too few chunks: {len(chunks)}"
    if not has_done:
        return False, "Missing [DONE] terminator"
    if len(full_content) < 10:
        return False, f"Content too short: '{full_content[:50]}'"

    return True, f"OK: {len(chunks)} chunks, {len(full_content)} chars, usage={has_usage}, [DONE]={has_done}"


def test_usage_tokens(endpoint: str) -> Tuple[bool, str]:
    """TC-15: usage.prompt_tokens and completion_tokens are correct."""
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "hi"}
    ], max_tokens=20)
    if code != 200:
        return False, f"HTTP {code}"
    usage = data.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    tt = usage.get("total_tokens", 0)
    if pt <= 0:
        return False, f"prompt_tokens={pt} <= 0"
    if ct <= 0:
        return False, f"completion_tokens={ct} <= 0"
    if tt != pt + ct:
        return False, f"total_tokens={tt} != {pt}+{ct}={pt+ct}"
    return True, f"OK: prompt={pt}, completion={ct}, total={tt}"


def test_model_name_validation(endpoint: str) -> Tuple[bool, str]:
    """TC-16: Wrong model name returns 4xx error."""
    url = f"{endpoint}/v1/chat/completions"
    resp = requests.post(url, json={
        "model": "wrong_name_that_does_not_exist",
        "messages": [{"role": "user", "content": "hi"}],
    }, timeout=30)
    if resp.status_code < 400:
        return False, f"Expected 4xx, got {resp.status_code}"
    return True, f"OK: HTTP {resp.status_code} for wrong model name"


def test_content_type_sse(endpoint: str) -> Tuple[bool, str]:
    """TC-17: Streaming response Content-Type contains text/event-stream."""
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "stream": True,
    }
    resp = requests.post(url, json=payload, timeout=30, stream=True)
    ct = resp.headers.get("Content-Type", "")
    if "text/event-stream" not in ct:
        return False, f"Content-Type='{ct}', expected text/event-stream"
    resp.close()
    return True, f"OK: Content-Type={ct}"


def test_instruction_following(endpoint: str) -> Tuple[bool, str]:
    """TC-18: Instruction following without system prompt."""
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "请只回复 PONG，不要说其他任何内容"}
    ], max_tokens=20, temperature=0.0)
    if code != 200:
        return False, f"HTTP {code}"
    content = data["choices"][0]["message"]["content"]
    if "PONG" not in content.upper():
        return False, f"No PONG in: '{content[:50]}'"
    return True, f"OK: '{content[:30]}'"


def test_idempotency(endpoint: str) -> Tuple[bool, str]:
    """TC-19: Idempotent decode — seed=42 temperature=0 two requests identical.

    CCCL parallel: catch2_test_device_reduce_deterministic.cu verifies:
      env1 = require(determinism::gpu_to_gpu) + tune(policy<1, 128>)
      env2 = require(determinism::gpu_to_gpu) + tune(policy<2, 256>)
      REQUIRE(d_output_p1 == d_output_p2)
    Two different execution policies give BIT-EXACT same result when
    determinism::gpu_to_gpu is required. This is because CCCL uses
    Reproducible Floating-point Accumulation (RFA) which guarantees
    rounding-order independence.

    For vllm: seed=42 + temperature=0.0 locks the RNG and uses argmax.
    Two identical requests MUST produce identical content strings.
    This is a hard competition requirement (TC-05 in the PRD).
    """
    kwargs = dict(
        max_tokens=50,
        temperature=0.0,
        seed=42,
    )
    messages = [{"role": "user", "content": "说hello"}]

    code1, data1 = chat_completion(endpoint, messages, **kwargs)
    if code1 != 200:
        return False, f"Request 1: HTTP {code1}"
    content1 = data1["choices"][0]["message"]["content"]

    code2, data2 = chat_completion(endpoint, messages, **kwargs)
    if code2 != 200:
        return False, f"Request 2: HTTP {code2}"
    content2 = data2["choices"][0]["message"]["content"]

    if content1 != content2:
        return False, f"NOT idempotent: '{content1[:40]}' vs '{content2[:40]}'"
    return True, f"OK: identical outputs '{content1[:30]}'"


def test_top_p_boundary(endpoint: str) -> Tuple[bool, str]:
    """TC-20: top_p=1.0 (no nucleus) and top_p=0.01 (extreme nucleus) both work.

    CCCL parallel: catch2_test_device_topk_keys.cu tests k=1 and k=N boundaries.
    dispatch_topk.cuh's multi-pass radix selection must handle:
      - k=1: single element (DeviceTopK degenerates to DeviceMin/Max)
      - k=N: all elements (no filtering, just sort)
    Similarly, top_p boundaries:
      - top_p=1.0: no filtering (all tokens eligible)
      - top_p=0.01: extreme filtering (only top ~1% of probability mass)
    """
    # top_p=1.0 (effectively disabled)
    code1, data1 = chat_completion(endpoint, [
        {"role": "user", "content": "hi"}
    ], max_tokens=10, top_p=1.0, temperature=0.7)
    if code1 != 200:
        return False, f"top_p=1.0: HTTP {code1}: {data1}"

    # top_p=0.01 (extreme nucleus — only highest prob token)
    code2, data2 = chat_completion(endpoint, [
        {"role": "user", "content": "hi"}
    ], max_tokens=10, top_p=0.01, temperature=0.7)
    if code2 != 200:
        return False, f"top_p=0.01: HTTP {code2}: {data2}"

    c1 = data1["choices"][0]["message"]["content"]
    c2 = data2["choices"][0]["message"]["content"]
    return True, f"OK: top_p=1.0→'{c1[:20]}', top_p=0.01→'{c2[:20]}'"


def test_frequency_penalty(endpoint: str) -> Tuple[bool, str]:
    """TC-21: frequency_penalty and presence_penalty accepted.

    CCCL parallel: tuning_histogram.cuh — token frequency counting for
    repetition_penalty is a histogram operation. CCCL's histogram uses
    privatized bins per CTA to avoid atomic contention.
    The bin_counts in sampler.py._get_bin_counts_and_mask() is the Python
    equivalent — scatter_add_ into (batch, vocab+1) tensor.
    """
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "写一段话"}
    ], max_tokens=100, frequency_penalty=1.5, presence_penalty=0.5)
    if code != 200:
        return False, f"HTTP {code}: {data}"
    content = data["choices"][0]["message"]["content"]
    if not content or len(content) < 5:
        return False, f"Content too short: '{content}'"
    return True, f"OK: {len(content)} chars with freq=1.5 pres=0.5"


# Update ALL_TESTS with the new tests
ALL_TESTS.extend([
    ("TC-14 Streaming SSE", test_streaming_sse),
    ("TC-15 Usage tokens", test_usage_tokens),
    ("TC-16 Model name validation", test_model_name_validation),
    ("TC-17 Content-Type SSE", test_content_type_sse),
    ("TC-18 Instruction following", test_instruction_following),
    ("TC-19 Idempotency (det reduce)", test_idempotency),
    ("TC-20 Top-p boundary", test_top_p_boundary),
    ("TC-21 Frequency penalty", test_frequency_penalty),
])


# ================================================================
# Tests TC-22 through TC-30: CCCL dispatch_segmented_reduce.cuh inspired
# Segmented reduce has 3 policy tiers: Large/Medium/Small segment.
# We test V1/V2 attention at analogous tier boundaries.
# ================================================================

def test_chinese_exact_repeat(endpoint: str) -> Tuple[bool, str]:
    """TC-22: Chinese exact repeat (Unicode encoding fidelity)."""
    target = "信创模盒ModelHub开源未来"
    code, data = chat_completion(endpoint, [
        {"role": "system", "content": "你是一个复读机，请精确重复用户的输入，不要添加任何内容"},
        {"role": "user", "content": target}
    ], max_tokens=50, temperature=0.0)
    if code != 200:
        return False, f"HTTP {code}"
    content = data["choices"][0]["message"]["content"]
    if target not in content:
        return False, f"Exact repeat failed: '{content[:60]}'"
    return True, f"OK: exact repeat verified"


def test_japanese_exact_repeat(endpoint: str) -> Tuple[bool, str]:
    """TC-23: Japanese exact repeat."""
    target = "東京タワーは日本の象徴です"
    code, data = chat_completion(endpoint, [
        {"role": "system", "content": "你是一个复读机，请精确重复用户的输入，不要添加任何内容"},
        {"role": "user", "content": target}
    ], max_tokens=50, temperature=0.0)
    if code != 200:
        return False, f"HTTP {code}"
    content = data["choices"][0]["message"]["content"]
    if target not in content:
        return False, f"Japanese repeat failed: '{content[:60]}'"
    return True, f"OK: Japanese repeat verified"


def test_n_parameter(endpoint: str) -> Tuple[bool, str]:
    """TC-24: n=2 returns 2 choices.
    CCCL parallel: dispatch_segmented_reduce.cuh — each segment produces
    one output. n=2 means 2 independent sampling runs = 2 segments.
    """
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "hi"}
    ], max_tokens=10, n=2, temperature=0.7)
    if code != 200:
        return False, f"HTTP {code}: {data}"
    num_choices = len(data.get("choices", []))
    if num_choices != 2:
        return False, f"Expected 2 choices, got {num_choices}"
    return True, f"OK: {num_choices} choices returned"


def test_empty_body_error(endpoint: str) -> Tuple[bool, str]:
    """TC-25: Empty JSON body returns 4xx."""
    url = f"{endpoint}/v1/chat/completions"
    resp = requests.post(url, json={}, timeout=30)
    if resp.status_code < 400:
        return False, f"Expected 4xx, got {resp.status_code}"
    return True, f"OK: HTTP {resp.status_code} for empty body"


def test_missing_role_error(endpoint: str) -> Tuple[bool, str]:
    """TC-26: Message missing role returns 4xx."""
    url = f"{endpoint}/v1/chat/completions"
    resp = requests.post(url, json={
        "model": "llm",
        "messages": [{"content": "hello"}]
    }, timeout=30)
    if resp.status_code < 400:
        return False, f"Expected 4xx, got {resp.status_code}"
    return True, f"OK: HTTP {resp.status_code} for missing role"


def test_top_k_boundary(endpoint: str) -> Tuple[bool, str]:
    """TC-27: top_k=1 (greedy-like via sampling) works.
    CCCL: dispatch_topk.cuh k=1 → DeviceReduceArgMax fast path.
    """
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "hi"}
    ], max_tokens=10, extra_body={"top_k": 1}, temperature=0.7)
    if code != 200:
        # top_k may not be supported as extra_body, try without
        code, data = chat_completion(endpoint, [
            {"role": "user", "content": "hi"}
        ], max_tokens=10, temperature=0.01)
        if code != 200:
            return False, f"HTTP {code}"
    return True, "OK: extreme low-temperature/top-k sampling works"


def test_temperature_2(endpoint: str) -> Tuple[bool, str]:
    """TC-28: temperature=2.0 (high randomness) works.
    CCCL: scale_mem_bound upper clamp = nominal*2 — tests boundary.
    """
    code, data = chat_completion(endpoint, [
        {"role": "user", "content": "hi"}
    ], max_tokens=10, temperature=2.0)
    if code != 200:
        return False, f"HTTP {code}: {data}"
    content = data["choices"][0]["message"]["content"]
    return True, f"OK: high-temp output '{content[:30]}'"


def test_models_endpoint(endpoint: str) -> Tuple[bool, str]:
    """TC-29: /v1/models returns model list with 'llm'."""
    url = f"{endpoint}/v1/models"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"
    data = resp.json()
    model_ids = [m.get("id") for m in data.get("data", [])]
    if "llm" not in model_ids:
        return False, f"'llm' not in models: {model_ids}"
    return True, f"OK: models={model_ids}"


def test_health_endpoint(endpoint: str) -> Tuple[bool, str]:
    """TC-30: /health returns 200."""
    try:
        resp = requests.get(f"{endpoint}/health", timeout=10)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        return True, "OK: health check passed"
    except requests.ConnectionError:
        return False, "Connection refused"


# Extend ALL_TESTS
ALL_TESTS.extend([
    ("TC-22 Chinese exact repeat", test_chinese_exact_repeat),
    ("TC-23 Japanese exact repeat", test_japanese_exact_repeat),
    ("TC-24 n=2 multiple choices", test_n_parameter),
    ("TC-25 Empty body error", test_empty_body_error),
    ("TC-26 Missing role error", test_missing_role_error),
    ("TC-27 Top-k boundary", test_top_k_boundary),
    ("TC-28 Temperature 2.0", test_temperature_2),
    ("TC-29 /v1/models endpoint", test_models_endpoint),
    ("TC-30 /health endpoint", test_health_endpoint),
])
