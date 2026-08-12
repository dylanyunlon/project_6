"""
Abstract reasoning parser base classes for vLLM 0.6.3.
Adapted from vllm-original/vllm/reasoning/abs_reasoning_parsers.py:
  - Removed vllm.entrypoints.mcp, vllm.utils.collection_utils, import_utils
  - DeltaMessage from vllm 0.6.3 protocol path
  - TokenizerLike -> AnyTokenizer
  - ReasoningParserManager: simplified eager + lazy registration
"""

import importlib
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.entrypoints.openai.protocol import DeltaMessage
    from vllm.transformers_utils.tokenizer import AnyTokenizer
else:
    DeltaMessage = Any
    AnyTokenizer = Any


class ReasoningParser:
    """Abstract base for all reasoning parsers."""

    def __init__(self, tokenizer: "AnyTokenizer", *args, **kwargs):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict:
        return self.model_tokenizer.get_vocab()

    @abstractmethod
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Return True once the reasoning block has closed in input_ids."""

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self.is_reasoning_end(input_ids)

    @abstractmethod
    def extract_content_ids(self, input_ids: list) -> list:
        """Return token ids that belong to the content (post-reasoning) part."""

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return 0

    @abstractmethod
    def extract_reasoning(
        self, model_output: str, request: Any
    ) -> "tuple[Optional[str], Optional[str]]":
        """
        Split a complete model output into (reasoning_text, content_text).
        Either part may be None.
        """

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Optional["DeltaMessage"]:
        """
        Extract reasoning from a streaming delta.
        Returns a DeltaMessage with reasoning_content and/or content set,
        or None if this delta should be suppressed (control token).
        """


class BaseThinkingReasoningParser(ReasoningParser):
    """
    Base for parsers that use <start_token>...</end_token> delimiters.
    Subclasses define start_token / end_token properties.
    """

    @property
    @abstractmethod
    def start_token(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def end_token(self) -> str:
        raise NotImplementedError

    def __init__(self, tokenizer: "AnyTokenizer", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError("Tokenizer must be passed to ReasoningParser.")
        if not self.start_token or not self.end_token:
            raise ValueError("start_token and end_token must be defined.")

        self.start_token_id: Optional[int] = self.vocab.get(self.start_token)
        self.end_token_id: Optional[int] = self.vocab.get(self.end_token)
        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: could not find think tokens "
                f"'{self.start_token}'/'{self.end_token}' in tokenizer vocab."
            )

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        for token_id in reversed(input_ids):
            if token_id == self.start_token_id:
                return False
            if token_id == self.end_token_id:
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self.end_token_id in delta_ids

    def extract_content_ids(self, input_ids: list) -> list:
        if self.end_token_id not in input_ids[:-1]:
            return []
        return input_ids[input_ids.index(self.end_token_id) + 1:]

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        count = 0
        depth = 0
        for tid in token_ids:
            if tid == self.start_token_id:
                depth += 1
            elif tid == self.end_token_id:
                if depth > 0:
                    depth -= 1
            elif depth > 0:
                count += 1
        return count

    def extract_reasoning(
        self, model_output: str, request: Any
    ) -> "tuple[Optional[str], Optional[str]]":
        # Strip <think> if the model generated it (old-style template).
        parts = model_output.partition(self.start_token)
        model_output = parts[2] if parts[1] else parts[0]

        if self.end_token not in model_output:
            return model_output, None
        reasoning, _, content = model_output.partition(self.end_token)
        return reasoning, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Optional["DeltaMessage"]:
        from vllm.entrypoints.openai.protocol import DeltaMessage as _DeltaMessage

        # Suppress lone control tokens.
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (
            self.start_token_id, self.end_token_id
        ):
            return None

        start_in_prev = self.start_token_id in previous_token_ids
        start_in_delta = self.start_token_id in delta_token_ids
        end_in_prev = self.end_token_id in previous_token_ids
        end_in_delta = self.end_token_id in delta_token_ids

        if start_in_prev:
            if end_in_delta:
                end_idx = delta_text.find(self.end_token)
                reasoning = delta_text[:end_idx] if end_idx >= 0 else ""
                content = delta_text[end_idx + len(self.end_token):] if end_idx >= 0 else None
                return _DeltaMessage(
                    reasoning_content=reasoning or None,
                    content=content or None,
                )
            elif end_in_prev:
                return _DeltaMessage(content=delta_text)
            else:
                return _DeltaMessage(reasoning_content=delta_text)

        elif start_in_delta:
            if end_in_delta:
                start_idx = delta_text.find(self.start_token)
                end_idx = delta_text.find(self.end_token)
                reasoning = delta_text[start_idx + len(self.start_token):end_idx]
                content = delta_text[end_idx + len(self.end_token):]
                return _DeltaMessage(
                    reasoning_content=reasoning or None,
                    content=content or None,
                )
            else:
                return _DeltaMessage(reasoning_content=delta_text)

        else:
            return _DeltaMessage(content=delta_text)


class ReasoningParserManager:
    """
    Registry for ReasoningParser implementations.
    Supports eager and lazy registration.
    """

    _parsers: dict = {}           # name -> class (eager)
    _lazy: dict = {}              # name -> (module_path, class_name)

    @classmethod
    def register_module(cls, name: str, parser_cls: type) -> None:
        """Eagerly register a ReasoningParser class."""
        if not issubclass(parser_cls, ReasoningParser):
            raise TypeError(f"{parser_cls} is not a ReasoningParser subclass.")
        cls._parsers[name] = parser_cls

    @classmethod
    def register_lazy(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a parser for deferred import."""
        cls._lazy[name] = (module_path, class_name)

    @classmethod
    def get_reasoning_parser(cls, name: str) -> type:
        if name in cls._parsers:
            return cls._parsers[name]
        if name in cls._lazy:
            module_path, class_name = cls._lazy[name]
            mod = importlib.import_module(module_path)
            parser_cls = getattr(mod, class_name)
            cls._parsers[name] = parser_cls
            return parser_cls
        registered = sorted(set(cls._parsers) | set(cls._lazy))
        raise KeyError(
            f"Reasoning parser '{name}' not found. "
            f"Available: {registered}"
        )

    @classmethod
    def list_registered(cls) -> list:
        return sorted(set(cls._parsers) | set(cls._lazy))
