"""In-process gRPC-web bridge.

The browser speaks gRPC-web (the gRPC framing over HTTP/1.1 POST bodies). This
ASGI endpoint decodes the request frame, calls the very same servicer object
the native gRPC server uses, and writes response frames plus gRPC trailers back
on the same HTTP port that serves the UI, so one process is enough.

Supported: unary and server-streaming calls, ``application/grpc-web+proto``
and ``application/grpc-web-text`` (base64, unary only).
"""

from __future__ import annotations

import base64
import inspect
import logging
import urllib.parse
from typing import Any, AsyncIterator

import grpc
from google.protobuf import symbol_database
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from .gen import urakatsu_pb2 as pb

log = logging.getLogger(__name__)

DATA_FRAME = 0x00
TRAILER_FRAME = 0x80
GRPC_WEB_PROTO = "application/grpc-web+proto"
GRPC_WEB_TEXT = "application/grpc-web-text"


class _Abort(Exception):
    def __init__(self, code: grpc.StatusCode, details: str):
        super().__init__(details)
        self.code = code
        self.details = details


class BridgeContext:
    """Minimal stand-in for ``grpc.aio.ServicerContext`` used by the servicer."""

    def __init__(self, headers):
        self._headers = headers
        self.code: grpc.StatusCode = grpc.StatusCode.OK
        self.details: str = ""

    async def abort(self, code: grpc.StatusCode, details: str = "") -> None:
        raise _Abort(code, details)

    def invocation_metadata(self):
        return tuple(self._headers.items())

    def set_code(self, code: grpc.StatusCode) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details

    async def send_initial_metadata(self, metadata) -> None:  # pragma: no cover - unused
        return None

    def peer(self) -> str:
        return "grpc-web"


def message_class(descriptor) -> type:
    """Resolves a message descriptor to its generated class across protobuf versions."""
    try:
        from google.protobuf import message_factory

        if hasattr(message_factory, "GetMessageClass"):
            return message_factory.GetMessageClass(descriptor)
    except ImportError:  # pragma: no cover
        pass
    database = symbol_database.Default()
    if hasattr(database, "GetPrototype"):
        return database.GetPrototype(descriptor)  # pragma: no cover - protobuf < 4
    return getattr(pb, descriptor.name)


def frame(flag: int, payload: bytes) -> bytes:
    return bytes([flag]) + len(payload).to_bytes(4, "big") + payload


def trailer(code: grpc.StatusCode, message: str = "") -> bytes:
    status = code.value[0]
    encoded = urllib.parse.quote(message, safe=" ") if message else ""
    text = f"grpc-status:{status}\r\n"
    if encoded:
        text += f"grpc-message:{encoded}\r\n"
    return frame(TRAILER_FRAME, text.encode("utf-8"))


def parse_frames(body: bytes) -> list[tuple[int, bytes]]:
    frames = []
    offset = 0
    while offset + 5 <= len(body):
        flag = body[offset]
        length = int.from_bytes(body[offset + 1 : offset + 5], "big")
        payload = body[offset + 5 : offset + 5 + length]
        frames.append((flag, payload))
        offset += 5 + length
    return frames


class GrpcWebBridge:
    def __init__(self, servicer: Any, service_name: str = "Urakatsu"):
        self.servicer = servicer
        descriptor = pb.DESCRIPTOR.services_by_name[service_name]
        self.full_name = descriptor.full_name
        self.methods: dict[str, tuple[Any, type, bool]] = {}
        for method in descriptor.methods:
            input_cls = message_class(method.input_type)
            self.methods[method.name] = (getattr(servicer, method.name), input_cls, method.server_streaming)

    @property
    def route(self) -> str:
        return f"/{self.full_name}/{{method}}"

    async def handle(self, request: Request) -> Response:
        if request.method == "OPTIONS":
            return Response(status_code=204)
        method_name = request.path_params.get("method", "")
        entry = self.methods.get(method_name)
        content_type = request.headers.get("content-type", GRPC_WEB_PROTO).split(";")[0].strip()
        text_mode = content_type == GRPC_WEB_TEXT
        if entry is None:
            return self._error(grpc.StatusCode.UNIMPLEMENTED, f"unknown method {method_name}", text_mode)
        handler, input_cls, streaming = entry

        body = await request.body()
        if text_mode:
            try:
                body = base64.b64decode(body)
            except ValueError:
                return self._error(grpc.StatusCode.INVALID_ARGUMENT, "invalid base64 body", text_mode)
        frames = parse_frames(body)
        payload = b""
        for flag, data in frames:
            if flag & TRAILER_FRAME == 0:
                payload = data
                break
        try:
            message = input_cls.FromString(payload)
        except Exception as error:
            return self._error(grpc.StatusCode.INVALID_ARGUMENT, f"could not decode request: {error}", text_mode)

        context = BridgeContext(request.headers)
        headers = {"content-type": GRPC_WEB_TEXT if text_mode else GRPC_WEB_PROTO, "cache-control": "no-cache"}
        if streaming:
            return StreamingResponse(self._stream(handler, message, context), headers=headers, media_type=headers["content-type"])
        try:
            result = await handler(message, context)
            out = frame(DATA_FRAME, result.SerializeToString()) + trailer(grpc.StatusCode.OK)
        except _Abort as abort:
            out = trailer(abort.code, abort.details)
        except Exception as error:
            log.exception("grpc-web %s failed", method_name)
            out = trailer(grpc.StatusCode.INTERNAL, str(error))
        if text_mode:
            out = base64.b64encode(out)
        return Response(content=out, headers=headers)

    async def _stream(self, handler, message, context) -> AsyncIterator[bytes]:
        try:
            iterator = handler(message, context)
            if inspect.isasyncgen(iterator):
                async for item in iterator:
                    yield frame(DATA_FRAME, item.SerializeToString())
            else:
                for item in await iterator:  # pragma: no cover - defensive
                    yield frame(DATA_FRAME, item.SerializeToString())
            yield trailer(grpc.StatusCode.OK)
        except _Abort as abort:
            yield trailer(abort.code, abort.details)
        except Exception as error:
            log.exception("grpc-web stream failed")
            yield trailer(grpc.StatusCode.INTERNAL, str(error))

    @staticmethod
    def _error(code: grpc.StatusCode, message: str, text_mode: bool) -> Response:
        out = trailer(code, message)
        if text_mode:
            out = base64.b64encode(out)
        return Response(content=out, media_type=GRPC_WEB_TEXT if text_mode else GRPC_WEB_PROTO)
