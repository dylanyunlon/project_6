"""
Reasoning parser for Qwen3 / Qwen3.5 / Qwen3.6 model family.
Adapted from vllm-original/vllm/reasoning/qwen3_reasoning_parser.py.

The model uses <think>...</think> to wrap chain-of-thought output.
For Qwen3.5+ the chat template injects <think> into the prompt, so only
</think> appears in the generated tokens; older templates generate <think>
themselves.  Both styles are handled.
"""

from typing import Optional, Sequence, Any

from vllm.reasoning.abs_reasoning_parsers import (
    BaseThinkingReasoningParser,
    ReasoningParserManager,
)


class Qwen3ReasoningParser(BaseThinkingReasoningParser):

    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning(
        self, model_output: str, request: Any
    ) -> "tuple[Optional[str], Optional[str]]":
        # Strip <think> if the model generated it (old template / edge case).
        parts = model_output.partition(self.start_token)
        model_output = parts[2] if parts[1] else parts[0]

        if not self.thinking_enabled:
            if self.end_token in model_output:
                _, _, content = model_output.partition(self.end_token)
                return None, content or ""
            return None, model_output

        if self.end_token not in model_output:
            # Thinking enabled but output truncated before </think>.
            return model_output, None

        reasoning, _, content = model_output.partition(self.end_token)
        return reasoning, content or None

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        token_ids = list(token_ids)
        if self.start_token_id in token_ids:
            # Old-style template: model generates <think> itself.
            # Use depth-counting from the base class.
            return super().count_reasoning_tokens(token_ids)
        elif self.end_token_id in token_ids:
            # New-style template (Qwen3.5+): <think> is injected into the
            # prompt, so output starts already inside the thinking block.
            # Every token before </think> is a reasoning token.
            return token_ids.index(self.end_token_id)
        else:
            # No </think> in output: either truncated (all reasoning)
            # or thinking disabled (none).
            return len(token_ids) if self.thinking_enabled else 0

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ):
        from vllm.entrypoints.openai.protocol import DeltaMessage

        if not self.thinking_enabled:
            return DeltaMessage(content=delta_text) if delta_text else None

        # Strip <think> from delta if the model generates it itself.
        if self.start_token_id in delta_token_ids:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token):]

        if self.end_token_id in delta_token_ids:
            end_idx = delta_text.find(self.end_token)
            if end_idx >= 0:
                reasoning = delta_text[:end_idx]
                content = delta_text[end_idx + len(self.end_token):]
                if not reasoning and not content:
                    return None
                return DeltaMessage(
                    reasoning_content=reasoning or None,
                    content=content or None,
                )
            return None

        if not delta_text:
            return None
        elif self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)
        else:
            return DeltaMessage(reasoning_content=delta_text)


# Register immediately when this module is imported.
ReasoningParserManager.register_module("qwen3", Qwen3ReasoningParser)
