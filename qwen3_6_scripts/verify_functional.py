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


def test_chat_dataset(endpoint: str) -> Tuple[bool, str]:
    """TC-11: Run chat_dataset_v0.json conversations."""
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
    ("TC-11 Chat dataset", test_chat_dataset),
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
