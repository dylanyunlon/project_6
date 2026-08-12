"""
Reasoning parser module for vLLM 0.6.3 (BI-V100 / Qwen3.6-35B-A3B adaptation).

Usage: --reasoning-parser qwen3
"""

from vllm.reasoning.abs_reasoning_parsers import ReasoningParser, ReasoningParserManager

__all__ = ["ReasoningParser", "ReasoningParserManager"]

# Lazy-register Qwen3 parser; imported on first get_reasoning_parser("qwen3").
ReasoningParserManager.register_lazy(
    "qwen3",
    "vllm.reasoning.qwen3_reasoning_parser",
    "Qwen3ReasoningParser",
)
