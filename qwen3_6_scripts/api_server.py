import asyncio
import importlib
import inspect
import multiprocessing
import os
import regex as re
import signal
import socket
import sys
import tempfile
import time
from argparse import Namespace
from contextlib import asynccontextmanager
from functools import partial
from http import HTTPStatus
from typing import AsyncIterator, Set


def _bi100_field(value, name):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _bi100_scalar(value):
    return getattr(value, "value", value)


def _bi100_tool_choice_kind(value):
    value = _bi100_scalar(value)
    if value is None:
        return "unset"
    if isinstance(value, str):
        return value if value in ("none", "auto", "required") else "other"
    function = _bi100_field(value, "function")
    if function is not None and isinstance(
            _bi100_field(function, "name"), str):
        return "named"
    return "other"


def _bi100_image_source_kind(value):
    if not isinstance(value, str):
        return "other"
    prefix = value[:8].lower()
    if prefix.startswith("data:"):
        return "data"
    if prefix.startswith(("http://", "https://")):
        return "remote"
    return "other"


def _bi100_chat_4xx_reason(message):
    if message == "messages must contain at least one message":
        return "empty_messages"
    if (isinstance(message, str)
            and message.startswith("top_p must be in (0, 1], got ")):
        return "invalid_top_p"
    if (isinstance(message, str)
            and message.startswith("max_tokens must be at least 1, got ")):
        return "invalid_max_tokens"
    if (isinstance(message, str)
            and message.startswith("This model's maximum context length is ")
            and "tokens. However, you requested " in message):
        return "context_length_exceeded"
    if (isinstance(message, str) and message.startswith("n=")
            and " exceeds max_num_seqs=" in message):
        return "n_exceeds_max_num_seqs"
    if message == 'tool_choice = "required" is not supported!':
        return "unsupported_tool_choice_required"
    if (isinstance(message, str)
            and message.startswith('"auto" tool choice requires ')):
        return "tool_parser_unavailable"
    if message == "Tool call arguments are not valid JSON.":
        return "invalid_tool_arguments_json"
    if (isinstance(message, str)
            and message.startswith("Tool call arguments must ")):
        return "invalid_tool_arguments_type"
    if (isinstance(message, str)
            and (
                (message.startswith("At most ")
                 and " image(s) may be provided in one request." in message)
                or (message.startswith("You set image=")
                    and "items in the same prompt." in message))):
        return "image_count_limit"
    if message == "Unknown model type: qwen3_5_moe":
        return "image_model_type_unsupported"
    return "unclassified_chat_error"


def _bi100_chat_request_shape(request):
    messages = _bi100_field(request, "messages")
    if not isinstance(messages, (list, tuple)):
        messages = ()
    tools = _bi100_field(request, "tools")
    if not isinstance(tools, (list, tuple)):
        tools = ()

    system_count = 0
    system_part_message_count = 0
    system_text_part_count = 0
    system_other_part_count = 0
    tool_message_count = 0
    assistant_tool_message_count = 0
    image_count = 0
    image_data_count = 0
    image_remote_count = 0
    image_other_count = 0
    for message in messages:
        role = _bi100_scalar(_bi100_field(message, "role"))
        if role == "system":
            system_count += 1
        elif role == "tool":
            tool_message_count += 1
        elif (role == "assistant"
              and _bi100_field(message, "tool_calls")):
            assistant_tool_message_count += 1
        content = _bi100_field(message, "content")
        if not isinstance(content, (list, tuple)):
            continue
        if role == "system":
            system_part_message_count += 1
        for part in content:
            part_type = _bi100_scalar(_bi100_field(part, "type"))
            if role == "system":
                if part_type == "text":
                    system_text_part_count += 1
                else:
                    system_other_part_count += 1
            if part_type in ("image", "image_url"):
                image_count += 1
                image_url = _bi100_field(part, "image_url")
                source_kind = _bi100_image_source_kind(
                    _bi100_field(image_url, "url"))
                if source_kind == "data":
                    image_data_count += 1
                elif source_kind == "remote":
                    image_remote_count += 1
                else:
                    image_other_count += 1

    strict_false_count = 0
    strict_true_count = 0
    for tool in tools:
        function = _bi100_field(tool, "function")
        strict = _bi100_field(function, "strict")
        if strict is False:
            strict_false_count += 1
        elif strict is True:
            strict_true_count += 1

    n = _bi100_field(request, "n")
    return {
        "message_count": len(messages),
        "system_count": system_count,
        "system_part_message_count": system_part_message_count,
        "system_text_part_count": system_text_part_count,
        "system_other_part_count": system_other_part_count,
        "tool_count": len(tools),
        "tool_message_count": tool_message_count,
        "assistant_tool_message_count": assistant_tool_message_count,
        "strict_false_count": strict_false_count,
        "strict_true_count": strict_true_count,
        "tool_choice_kind": _bi100_tool_choice_kind(
            _bi100_field(request, "tool_choice")),
        "image_count": image_count,
        "image_data_count": image_data_count,
        "image_remote_count": image_remote_count,
        "image_other_count": image_other_count,
        "has_image": image_count > 0,
        "stream": bool(_bi100_field(request, "stream")),
        "n": n if isinstance(n, int) else None,
    }


def _bi100_validation_message_reason(error, tool_choice_kind):
    if not isinstance(error, dict):
        return None

    messages = []
    context = error.get("ctx")
    if isinstance(context, dict):
        context_error = context.get("error")
        if isinstance(context_error, ValueError):
            messages.append(str(context_error))

    message = error.get("msg")
    if isinstance(message, str):
        if message.startswith("Value error, "):
            message = message.removeprefix("Value error, ")
        messages.append(message)

    for message in messages:
        if message == "Tool call arguments are not valid JSON.":
            return "invalid_tool_arguments_json"
        if message in (
                "Tool call arguments must decode to a JSON object.",
                "Tool call arguments must be a JSON object or a "
                "JSON-encoded object string."):
            return "invalid_tool_arguments_type"
        if message == (
                "`tool_choice` must be a named tool, \"auto\", or \"none\"."):
            if tool_choice_kind == "required":
                return "unsupported_tool_choice_required"
            return "request_validation_tool_choice"
    return None


def _bi100_validation_reason(errors, request_shape=None):
    categories = set()
    message_categories = set()
    tool_choice_kind = (
        request_shape.get("tool_choice_kind")
        if isinstance(request_shape, dict) else None
    )
    validation_errors = errors if isinstance(errors, (list, tuple)) else ()
    for error in validation_errors:
        if not isinstance(error, dict):
            continue
        message_category = _bi100_validation_message_reason(
            error, tool_choice_kind)
        if message_category is not None:
            message_categories.add(message_category)
        location = error.get("loc")
        if not isinstance(location, (list, tuple)):
            continue
        fields = [
            value for value in location
            if isinstance(value, str)
            and value not in ("body", "query", "path")
        ]
        if not fields:
            continue
        field = fields[0]
        descendants = set(fields[1:])
        if field == "messages":
            if "tool_call_id" in descendants:
                categories.add("request_validation_message_tool_call_id")
            elif "tool_calls" in descendants:
                categories.add("request_validation_message_tool_calls")
            elif "content" in descendants:
                categories.add("request_validation_message_content")
            elif "role" in descendants:
                categories.add("request_validation_message_role")
            else:
                categories.add("request_validation_messages")
        elif field == "tools":
            if "strict" in descendants:
                categories.add("request_validation_tool_strict")
            elif "parameters" in descendants:
                categories.add("request_validation_tool_parameters")
            else:
                categories.add("request_validation_tools")
        elif field in ("tool_choice", "parallel_tool_calls"):
            categories.add("request_validation_tool_choice")
        elif field == "response_format":
            categories.add("request_validation_response_format")
        elif field in ("stream", "stream_options"):
            categories.add("request_validation_streaming")
        elif field in ("n", "max_tokens", "min_tokens", "stop"):
            categories.add("request_validation_generation")
        elif field in (
                "temperature", "top_p", "top_k", "frequency_penalty",
                "presence_penalty", "repetition_penalty", "seed"):
            categories.add("request_validation_sampling")
        elif field == "model":
            categories.add("request_validation_model")
        else:
            categories.add("request_validation_other")

    priority = (
        "request_validation_tool_strict",
        "request_validation_tool_parameters",
        "request_validation_tool_choice",
        "request_validation_message_tool_call_id",
        "request_validation_message_tool_calls",
        "request_validation_message_content",
        "request_validation_message_role",
        "request_validation_messages",
        "request_validation_tools",
        "request_validation_response_format",
        "request_validation_streaming",
        "request_validation_generation",
        "request_validation_sampling",
        "request_validation_model",
        "request_validation_other",
    )
    for category in priority:
        if category in categories:
            return category
    message_priority = (
        "invalid_tool_arguments_json",
        "invalid_tool_arguments_type",
        "unsupported_tool_choice_required",
        "request_validation_tool_choice",
    )
    for category in message_priority:
        if category in message_categories:
            return category
    return "request_validation_unknown"


def _bi100_validation_identifier(value):
    if not isinstance(value, str) or not value or len(value) > 64:
        return "unknown"
    if not value.isascii():
        return "unknown"
    if not all(character.isalnum() or character in "._-"
               for character in value):
        return "unknown"
    return value


def _bi100_validation_diagnostics(errors):
    if not isinstance(errors, (list, tuple)):
        return "unknown", "unknown"
    try:
        error_count = len(errors)
    except Exception:
        return "unknown", "unknown"
    if error_count > 1:
        return "multiple", "multiple"
    if error_count == 0:
        return "unknown", "unknown"

    try:
        error = errors[0]
        if not isinstance(error, dict):
            return "unknown", "unknown"
        location = error.get("loc")
        validation_type = _bi100_validation_identifier(error.get("type"))
        if not isinstance(location, (list, tuple)):
            return "unknown", validation_type
        if not location:
            return "root", validation_type
        index = 0
        if location[0] in ("body", "query", "path", "header", "cookie"):
            index = 1
        if index >= len(location):
            return "root", validation_type
        field = location[index]
        if field in ("__root__", "root"):
            return "root", validation_type
        return _bi100_validation_identifier(field), validation_type
    except Exception:
        return "unknown", "unknown"


def _bi100_safe_validation_errors(exc):
    try:
        errors = exc.errors()
        if not isinstance(errors, (list, tuple)):
            return ()
        return tuple(errors)
    except Exception:
        return ()


def _bi100_startup_trace(message: str) -> None:
    if os.getenv("BI100_EXECUTOR_STARTUP_DEBUG") == "1":
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"[BI100 STARTUP] {stamp} pid={os.getpid()} {message}",
              file=sys.stderr, flush=True)


_bi100_startup_trace("api_server stdlib imports complete; loading runtime dependencies")

import uvloop
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import State
from starlette.routing import Mount
from typing_extensions import assert_never

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              CompletionResponse,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              EmbeddingRequest,
                                              EmbeddingResponse, ErrorResponse,
                                              LoadLoraAdapterRequest,
                                              TokenizeRequest,
                                              TokenizeResponse,
                                              UnloadLoraAdapterRequest)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_engine import BaseModelPath
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, get_open_zmq_ipc_path
from vllm.version import __version__ as VLLM_VERSION

TIMEOUT_KEEP_ALIVE = 5  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()

_bi100_startup_trace("api_server runtime imports complete")


def _bi100_log_chat_4xx(request, error) -> None:
    code = getattr(error, "code", None)
    if not isinstance(code, int) or not 400 <= code < 500:
        return
    shape = _bi100_chat_request_shape(request)
    reason = _bi100_chat_4xx_reason(getattr(error, "message", None))
    logger.warning(
        "[BI100 4XX] endpoint=chat code=%d reason=%s messages=%d "
        "systems=%d system_part_msgs=%d system_text_parts=%d "
        "system_other_parts=%d tools=%d tool_msgs=%d "
        "assistant_tool_msgs=%d strict_false=%d strict_true=%d choice=%s "
        "images=%d image_data=%d image_remote=%d image_other=%d "
        "stream=%d n=%s",
        code,
        reason,
        shape["message_count"],
        shape["system_count"],
        shape["system_part_message_count"],
        shape["system_text_part_count"],
        shape["system_other_part_count"],
        shape["tool_count"],
        shape["tool_message_count"],
        shape["assistant_tool_message_count"],
        shape["strict_false_count"],
        shape["strict_true_count"],
        shape["tool_choice_kind"],
        shape["image_count"],
        shape["image_data_count"],
        shape["image_remote_count"],
        shape["image_other_count"],
        int(shape["stream"]),
        shape["n"] if shape["n"] is not None else "unset",
    )


def _bi100_log_request_validation_4xx(raw_request, exc) -> None:
    validation_errors = ()
    validation_field = "unknown"
    validation_type = "unknown"
    try:
        validation_errors = _bi100_safe_validation_errors(exc)
        validation_field, validation_type = (
            _bi100_validation_diagnostics(validation_errors)
        )
        body = getattr(exc, "body", None)
        url = getattr(raw_request, "url", None)
        path = getattr(url, "path", "")
        is_chat_request = (
            isinstance(path, str)
            and path.endswith("/v1/chat/completions")
            and isinstance(body, dict)
        )
        shape = (
            _bi100_chat_request_shape(body) if is_chat_request else None
        )
        reason = _bi100_validation_reason(validation_errors, shape)
        if shape is not None:
            if (reason == "request_validation_tools"
                    and shape["strict_true_count"]):
                reason = "request_validation_tool_strict"
            logger.warning(
                "[BI100 4XX] endpoint=request_validation code=400 reason=%s "
                "messages=%d systems=%d system_part_msgs=%d "
                "system_text_parts=%d system_other_parts=%d tools=%d "
                "tool_msgs=%d assistant_tool_msgs=%d strict_false=%d "
                "strict_true=%d choice=%s images=%d image_data=%d "
                "image_remote=%d image_other=%d stream=%d n=%s errors=%d "
                "validation_field=%s validation_type=%s",
                reason,
                shape["message_count"],
                shape["system_count"],
                shape["system_part_message_count"],
                shape["system_text_part_count"],
                shape["system_other_part_count"],
                shape["tool_count"],
                shape["tool_message_count"],
                shape["assistant_tool_message_count"],
                shape["strict_false_count"],
                shape["strict_true_count"],
                shape["tool_choice_kind"],
                shape["image_count"],
                shape["image_data_count"],
                shape["image_remote_count"],
                shape["image_other_count"],
                int(shape["stream"]),
                shape["n"] if shape["n"] is not None else "unset",
                len(validation_errors),
                validation_field,
                validation_type,
            )
        else:
            logger.warning(
                "[BI100 4XX] endpoint=request_validation code=400 reason=%s "
                "errors=%d validation_field=%s validation_type=%s",
                reason,
                len(validation_errors),
                validation_field,
                validation_type,
            )
        return
    except Exception:
        pass

    try:
        logger.warning(
            "[BI100 4XX] endpoint=request_validation code=400 "
            "reason=request_validation_unknown errors=%d "
            "validation_field=%s validation_type=%s",
            len(validation_errors),
            validation_field,
            validation_type,
        )
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client

            async def _force_log():
                while True:
                    await asyncio.sleep(10.)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)
        else:
            task = None
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state


@asynccontextmanager
async def build_async_engine_client(
        args: Namespace) -> AsyncIterator[EngineClient]:

    _bi100_startup_trace("building AsyncEngineArgs")
    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)

    _bi100_startup_trace("entering engine client construction")
    async with build_async_engine_client_from_engine_args(
            engine_args, args.disable_frontend_multiprocessing) as engine:
        _bi100_startup_trace("engine client construction completed")
        yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Fall back
    # TODO: fill out feature matrix.
    if (MQLLMEngineClient.is_unsupported_config(engine_args)
            or disable_frontend_multiprocessing):
        engine_config = engine_args.create_engine_config()
        uses_ray = getattr(AsyncLLMEngine._get_executor_cls(engine_config),
                           "uses_ray", False)

        build_engine = partial(AsyncLLMEngine.from_engine_args,
                               engine_args=engine_args,
                               engine_config=engine_config,
                               usage_context=UsageContext.OPENAI_API_SERVER)
        if uses_ray:
            # Must run in main thread with ray for its signal handlers to work
            engine_client = build_engine()
        else:
            engine_client = await asyncio.get_running_loop().run_in_executor(
                None, build_engine)

        yield engine_client
        return

    # Otherwise, use the multiprocessing AsyncLLMEngine.
    else:
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            # Make TemporaryDirectory for prometheus multiprocessing
            # Note: global TemporaryDirectory will be automatically
            #   cleaned up upon exit.
            global prometheus_multiproc_dir
            prometheus_multiproc_dir = tempfile.TemporaryDirectory()
            os.environ[
                "PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
        else:
            logger.warning(
                "Found PROMETHEUS_MULTIPROC_DIR was set by user. "
                "This directory must be wiped between vLLM runs or "
                "you will find inaccurate metrics. Unset the variable "
                "and vLLM will properly handle cleanup.")

        # Select random path for IPC.
        ipc_path = get_open_zmq_ipc_path()
        logger.info("Multiprocessing frontend to use %s for IPC Path.",
                    ipc_path)

        # Start RPCServer in separate process (holds the LLMEngine).
        # the current process might have CUDA context,
        # so we need to spawn a new process
        context = multiprocessing.get_context("spawn")

        engine_process = context.Process(target=run_mp_engine,
                                         args=(engine_args,
                                               UsageContext.OPENAI_API_SERVER,
                                               ipc_path))
        engine_process.start()
        logger.info("Started engine process with PID %d", engine_process.pid)

        # Build RPCClient, which conforms to EngineClient Protocol.
        # NOTE: Actually, this is not true yet. We still need to support
        # embedding models via RPC (see TODO above)
        engine_config = engine_args.create_engine_config()
        mp_engine_client = MQLLMEngineClient(ipc_path, engine_config)

        try:
            while True:
                try:
                    await mp_engine_client.setup()
                    break
                except TimeoutError:
                    if not engine_process.is_alive():
                        raise RuntimeError(
                            "Engine process failed to start") from None

            yield mp_engine_client  # type: ignore[misc]
        finally:
            # Ensure rpc server process was terminated
            engine_process.terminate()

            # Close all open connections to the backend
            mp_engine_client.close()

            # Wait for engine process to join
            engine_process.join(4)
            if engine_process.exitcode is None:
                # Kill if taking longer than 5 seconds to stop
                engine_process.kill()

            # Lazy import for prometheus multiprocessing.
            # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
            # before prometheus_client is imported.
            # See https://prometheus.github.io/client_python/multiprocess/
            from prometheus_client import multiprocess
            multiprocess.mark_process_dead(engine_process.pid)


router = APIRouter()


def mount_metrics(app: FastAPI):
    # Lazy import for prometheus multiprocessing.
    # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
    # before prometheus_client is imported.
    # See https://prometheus.github.io/client_python/multiprocess/
    from prometheus_client import (CollectorRegistry, make_asgi_app,
                                   multiprocess)

    prometheus_multiproc_dir_path = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
    if prometheus_multiproc_dir_path is not None:
        logger.info("vLLM to use %s as PROMETHEUS_MULTIPROC_DIR",
                    prometheus_multiproc_dir_path)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)

        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app(registry=registry))
    else:
        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app())

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


def chat(request: Request) -> OpenAIServingChat:
    return request.app.state.openai_serving_chat


def completion(request: Request) -> OpenAIServingCompletion:
    return request.app.state.openai_serving_completion


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


def embedding(request: Request) -> OpenAIServingEmbedding:
    return request.app.state.openai_serving_embedding


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    await engine_client(raw_request).check_health()
    return Response(status_code=200)


@router.post("/tokenize")
async def tokenize(request: TokenizeRequest, raw_request: Request):
    generator = await tokenization(raw_request).create_tokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, TokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/detokenize")
async def detokenize(request: DetokenizeRequest, raw_request: Request):
    generator = await tokenization(raw_request).create_detokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, DetokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    models = await completion(raw_request).show_available_models()
    return JSONResponse(content=models.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):

    generator = await chat(raw_request).create_chat_completion(
        request, raw_request)

    if isinstance(generator, ErrorResponse):
        _bi100_log_chat_4xx(request, generator)
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await completion(raw_request).create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    generator = await embedding(raw_request).create_embedding(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, EmbeddingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


if envs.VLLM_TORCH_PROFILER_DIR:
    logger.warning(
        "Torch Profiler is enabled in the API server. This should ONLY be "
        "used for local development!")

    @router.post("/start_profile")
    async def start_profile(raw_request: Request):
        logger.info("Starting profiler...")
        await engine_client(raw_request).start_profile()
        logger.info("Profiler started.")
        return Response(status_code=200)

    @router.post("/stop_profile")
    async def stop_profile(raw_request: Request):
        logger.info("Stopping profiler...")
        await engine_client(raw_request).stop_profile()
        logger.info("Profiler stopped.")
        return Response(status_code=200)


if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
    logger.warning(
        "Lora dynamic loading & unloading is enabled in the API server. "
        "This should ONLY be used for local development!")

    @router.post("/v1/load_lora_adapter")
    async def load_lora_adapter(request: LoadLoraAdapterRequest,
                                raw_request: Request):
        response = await chat(raw_request).load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        response = await completion(raw_request).load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)

    @router.post("/v1/unload_lora_adapter")
    async def unload_lora_adapter(request: UnloadLoraAdapterRequest,
                                  raw_request: Request):
        response = await chat(raw_request).unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        response = await completion(raw_request).unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)


def build_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(openapi_url=None,
                      docs_url=None,
                      redoc_url=None,
                      lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(raw_request, exc):
        _bi100_log_request_validation_4xx(raw_request, exc)
        chat = app.state.openai_serving_chat
        err = chat.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    return app


def init_app_state(
    engine_client: EngineClient,
    model_config: ModelConfig,
    state: State,
    args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats

    state.openai_serving_chat = OpenAIServingChat(
        engine_client,
        model_config,
        base_model_paths,
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=args.chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
        reasoning_parser=getattr(args, 'reasoning_parser', None))
    state.openai_serving_completion = OpenAIServingCompletion(
        engine_client,
        model_config,
        base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )
    state.openai_serving_embedding = OpenAIServingEmbedding(
        engine_client,
        model_config,
        base_model_paths,
        request_logger=request_logger,
    )
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        model_config,
        base_model_paths,
        lora_modules=args.lora_modules,
        request_logger=request_logger,
        chat_template=args.chat_template,
    )


async def run_server(args, **uvicorn_kwargs) -> None:
    _bi100_startup_trace("run_server entered")
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valide_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice \
        and args.tool_call_parser not in valide_tool_parses:
        raise KeyError(f"invalid tool call parser: {args.tool_call_parser} "
                       f"(chose from {{ {','.join(valide_tool_parses)} }})")

    reasoning_parser = getattr(args, 'reasoning_parser', None)
    if reasoning_parser:
        valid_reasoning = ReasoningParserManager.list_registered()
        if reasoning_parser not in valid_reasoning:
            raise KeyError(
                f"invalid reasoning parser: {reasoning_parser} "
                f"(chose from {{ {','.join(valid_reasoning)} }})")

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", args.port))

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    _bi100_startup_trace("starting engine client context")
    async with build_async_engine_client(args) as engine_client:
        _bi100_startup_trace("building FastAPI application")
        app = build_app(args)

        _bi100_startup_trace("requesting model config from engine")
        model_config = await engine_client.get_model_config()
        _bi100_startup_trace("model config received; initializing app state")
        init_app_state(engine_client, model_config, app.state, args)

        _bi100_startup_trace("starting HTTP server")
        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            fd=sock.fileno(),
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task


if __name__ == "__main__":
    _bi100_startup_trace("api_server __main__ entered")
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    _bi100_startup_trace(
        f"arguments parsed model={args.model} tp={args.tensor_parallel_size} "
        f"max_model_len={args.max_model_len}")

    uvloop.run(run_server(args))
