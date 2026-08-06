from .abstract_tool_parser import ToolParser, ToolParserManager
from .hermes_tool_parser import Hermes2ProToolParser
from .internlm2_tool_parser import Internlm2ToolParser
from .llama_tool_parser import Llama3JsonToolParser
from .mistral_tool_parser import MistralToolParser

# Register qwen3_coder as alias for hermes parser.
# Qwen3 models use Hermes-compatible tool calling format:
#   <tool_call>{"name": "func", "arguments": {...}}</tool_call>
# computility-run.yaml specifies --tool-call-parser qwen3_coder
# which must be registered or server startup crashes with KeyError.
ToolParserManager.register_module(
    "qwen3_coder", module=Hermes2ProToolParser)

__all__ = [
    "ToolParser", "ToolParserManager", "Hermes2ProToolParser",
    "MistralToolParser", "Internlm2ToolParser", "Llama3JsonToolParser"
]
