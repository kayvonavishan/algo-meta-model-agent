from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Tracer


_TRACER: Optional[Tracer] = None
_TRACER_PROVIDER = None


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _max_text_chars() -> int:
    try:
        return int(os.getenv("PHOENIX_MAX_TEXT_CHARS", "200000"))
    except Exception:
        return 200000


def _truncate(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    max_chars = _max_text_chars()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n... [truncated]\n"


def init_phoenix_tracing(*, project_name: Optional[str] = None, endpoint: Optional[str] = None) -> Tracer:
    """
    Initialize OpenTelemetry tracing for Arize Phoenix and (optionally) instrument OpenAI Agents.

    Environment variables:
    - PHOENIX_COLLECTOR_ENDPOINT: defaults to http://localhost:6006
    - PHOENIX_PROJECT_NAME: defaults to "algo-meta-model-agent"
    - PHOENIX_MAX_TEXT_CHARS: truncation limit for stored text attributes (default 200000)
    - PHOENIX_LOG_CONTENT: if true, attach prompt/output text into spans (default true)
    """
    global _TRACER, _TRACER_PROVIDER  # pylint: disable=global-statement

    if _TRACER is not None:
        return _TRACER

    project_name = project_name or os.getenv("PHOENIX_PROJECT_NAME") or "algo-meta-model-agent"

    # `PHOENIX_COLLECTOR_ENDPOINT` is commonly set to the Phoenix *app* URL (UI),
    # e.g. http://localhost:6006. The OTEL trace receiver is typically:
    # - HTTP/protobuf: http://localhost:6006/v1/traces
    # - gRPC:          http://localhost:4317
    # See Phoenix docs for default ports/endpoints.
    phoenix_host = endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or "http://localhost:6006"
    otel_endpoint = os.getenv("PHOENIX_OTEL_ENDPOINT") or os.getenv("PHOENIX_TRACING_ENDPOINT") or ""
    otel_endpoint = (otel_endpoint or phoenix_host).strip()

    def _normalize_url(raw: str) -> str:
        s = (raw or "").strip()
        if not s:
            return ""
        if "://" not in s:
            s = "http://" + s
        return s

    def _ensure_http_traces_path(raw: str) -> str:
        raw = _normalize_url(raw)
        u = urlparse(raw)
        path = (u.path or "").rstrip("/")
        if path in ("", "/"):
            path = "/v1/traces"
        return urlunparse(u._replace(path=path))

    otel_endpoint = _normalize_url(otel_endpoint)
    parsed = urlparse(otel_endpoint)
    protocol = None
    # Heuristics:
    # - If user provided /v1/traces or port 6006, assume HTTP/protobuf.
    # - If port 4317, assume gRPC.
    if parsed.port == 4317:
        protocol = "grpc"
        # gRPC exporter expects host:port, without /v1/traces.
        otel_endpoint = urlunparse(parsed._replace(path=""))
    else:
        if parsed.path.rstrip("/") == "/v1/traces" or parsed.port == 6006 or parsed.path.strip() == "":
            protocol = "http/protobuf"
            otel_endpoint = _ensure_http_traces_path(otel_endpoint)

    try:
        from phoenix.otel import register  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Phoenix OTEL package not available. Install with: pip install arize-phoenix-otel"
        ) from exc

    register_attempts = []
    if otel_endpoint:
        kwargs = {"project_name": project_name, "endpoint": otel_endpoint}
        if protocol:
            kwargs["protocol"] = protocol
        register_attempts.append(kwargs)
    register_attempts.append({"project_name": project_name})
    last_exc: Optional[Exception] = None
    for kwargs in register_attempts:
        try:
            _TRACER_PROVIDER = register(**kwargs)
            last_exc = None
            break
        except TypeError as exc:  # pragma: no cover
            last_exc = exc
            continue
    if _TRACER_PROVIDER is None:  # pragma: no cover
        raise RuntimeError(f"Phoenix OTEL register failed: {last_exc}") from last_exc
    _TRACER = trace.get_tracer("algo-meta-model-agent")

    # Best-effort: instrument OpenAI Agents SDK so LLM/tool spans appear automatically.
    try:
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor  # type: ignore

        OpenAIAgentsInstrumentor().instrument(tracer_provider=_TRACER_PROVIDER)
    except Exception:
        # Optional dependency or incompatible versions; keep manual spans working.
        pass

    return _TRACER


def get_tracer() -> Tracer:
    if _TRACER is None:
        return init_phoenix_tracing()
    return _TRACER


def phoenix_log_content_enabled(default: bool = True) -> bool:
    return _env_bool("PHOENIX_LOG_CONTENT", default=default)


def set_openinference_kind(span, kind: str) -> None:  # noqa: ANN001
    try:
        span.set_attribute("openinference.span.kind", str(kind))
    except Exception:
        return


def set_attrs(span, attrs: Optional[Dict[str, Any]] = None) -> None:  # noqa: ANN001
    if not attrs:
        return
    for k, v in attrs.items():
        try:
            if v is None:
                continue
            span.set_attribute(str(k), v if isinstance(v, (str, int, float, bool)) else str(v))
        except Exception:
            continue


def set_text(span, key: str, text: str) -> None:  # noqa: ANN001
    try:
        span.set_attribute(str(key), _truncate(text))
    except Exception:
        return


def set_io(span, *, input_text: Optional[str] = None, output_text: Optional[str] = None) -> None:  # noqa: ANN001
    if not phoenix_log_content_enabled(default=True):
        return
    if input_text is not None:
        set_text(span, "input.value", input_text)
    if output_text is not None:
        set_text(span, "output.value", output_text)
