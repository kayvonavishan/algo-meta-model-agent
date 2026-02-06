import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


class LLMClient:
    def generate(self, prompt, system_prompt=None):
        raise NotImplementedError


class DummyClient(LLMClient):
    def __init__(self, config):
        self.config = config

    def generate(self, prompt, system_prompt=None):
        prompt_lower = prompt.lower()
        if "unified diff" in prompt_lower or "patch" in prompt_lower:
            mode = self.config.get("dummy_patch_mode", "noop")
            if mode == "sample_patch":
                path = self.config.get("dummy_patch_path", "")
                if path and os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return f.read()
            return ""
        result = self.config.get(
            "dummy_idea_text",
            "IDEA: Increase top_n_global by a small amount to test sensitivity.\n"
            "RATIONALE: Tests robustness to selection breadth.\n"
            "FILES: adaptive_vol_momentum.py\n"
            "RISKS: Might dilute signal if too large.",
        )
        if _should_log(self.config):
            _log_llm_call("dummy", self.config.get("model", "dummy"), prompt, result)
        return result


class OpenAIClient(LLMClient):
    def __init__(self, config):
        self.config = dict(config or {})
        self.config.setdefault("debug_log", True)
        self.api_key = os.getenv(self.config.get("api_key_env", "OPENAI_API_KEY"), "")
        self.base_url = self.config.get("base_url") or "https://api.openai.com/v1"

    def generate(self, prompt, system_prompt=None):
        if not self.api_key:
            raise RuntimeError("OpenAI API key env var not set.")
        url = f"{self.base_url}/responses"
        inputs = []
        if system_prompt:
            inputs.append({"role": "system", "content": system_prompt})
        inputs.append({"role": "user", "content": prompt})
        model = self.config.get("model", "gpt-4.1")
        payload = {
            "model": model,
            "input": inputs,
            "max_output_tokens": self.config.get("max_tokens", 1200),
        }
        reasoning = self.config.get("reasoning")
        if reasoning:
            payload["reasoning"] = {"effort": reasoning}
        temperature = self.config.get("temperature", None)
        if temperature is not None and _openai_model_supports_temperature(model):
            payload["temperature"] = temperature
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if _should_log(self.config):
            _ensure_default_debug_file("openai")
        result = _post_json_openai_with_fallback(
            url,
            payload,
            headers,
            self.config.get("timeout_sec", 60),
            debug_meta={"provider": "openai", "model": model, "config": self.config},
        )
        if _should_log(self.config):
            _log_llm_call("openai", payload.get("model"), prompt, result)
        return result


class AnthropicClient(LLMClient):
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv(config.get("api_key_env", "ANTHROPIC_API_KEY"), "")
        self.base_url = config.get("base_url") or "https://api.anthropic.com"

    def generate(self, prompt, system_prompt=None):
        if not self.api_key:
            raise RuntimeError("Anthropic API key env var not set.")
        url = f"{self.base_url}/v1/messages"
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.config.get("model", "claude-3-5-sonnet-20241022"),
            "max_tokens": self.config.get("max_tokens", 1200),
            "temperature": self.config.get("temperature", 0.2),
            "messages": messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        reasoning = self.config.get("reasoning")
        if reasoning:
            payload.setdefault("metadata", {})
            payload["metadata"]["reasoning"] = reasoning
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        result = _post_json(
            url,
            payload,
            headers,
            self.config.get("timeout_sec", 60),
            debug_meta={"provider": "anthropic", "model": payload.get("model"), "config": self.config},
        )
        if _should_log(self.config):
            _log_llm_call("anthropic", payload.get("model"), prompt, result)
        return result


class GeminiClient(LLMClient):
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv(config.get("api_key_env", "GEMINI_API_KEY"), "")
        self.base_url = config.get("base_url") or "https://generativelanguage.googleapis.com"

    def generate(self, prompt, system_prompt=None):
        if not self.api_key:
            raise RuntimeError("Gemini API key env var not set.")
        model = self.config.get("model", "gemini-1.5-pro")
        url = f"{self.base_url}/v1beta/models/{model}:generateContent?key={self.api_key}"
        parts = [{"text": prompt}]
        if system_prompt:
            parts.insert(0, {"text": system_prompt})
        payload = {"contents": [{"parts": parts}]}
        reasoning = self.config.get("reasoning")
        if reasoning:
            payload.setdefault("safetySettings", [])
            payload["metadata"] = {"reasoning": reasoning}
        headers = {"content-type": "application/json"}
        result = _post_json(
            url,
            payload,
            headers,
            self.config.get("timeout_sec", 60),
            debug_meta={"provider": "gemini", "model": model, "config": self.config},
        )
        if _should_log(self.config):
            _log_llm_call("gemini", model, prompt, result)
        return result


def build_llm_client(config):
    provider = (config.get("provider") or "dummy").lower()
    if provider in ("openai", "oai"):
        return OpenAIClient(config)
    if provider in ("anthropic", "claude"):
        return AnthropicClient(config)
    if provider in ("gemini", "google"):
        return GeminiClient(config)
    return DummyClient(config)


def _post_json(url, payload, headers, timeout_sec, debug_meta=None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status = getattr(resp, "status", None)
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        _maybe_log_http_response(debug_meta, url, exc.code, detail)
        raise RuntimeError(f"LLM request failed: {exc.code} {detail}") from exc

    _maybe_log_http_response(debug_meta, url, status, raw)
    parsed = json.loads(raw)
    return _extract_text(parsed)


def _openai_model_supports_temperature(model):
    m = (model or "").lower().strip()
    return not (m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"))


def _try_extract_unsupported_param(error_text):
    if not error_text:
        return None
    match = re.search(r"Unsupported parameter: '([^']+)'", error_text)
    if match:
        return match.group(1)
    match = re.search(r'"param"\\s*:\\s*"([^"]+)"', error_text)
    if match:
        return match.group(1)
    return None


def _post_json_openai_with_fallback(url, payload, headers, timeout_sec, debug_meta=None):
    removed = set()
    last_error = None
    for _ in range(3):
        try:
            return _post_json(url, payload, headers, timeout_sec, debug_meta=debug_meta)
        except RuntimeError as exc:
            last_error = str(exc)
            param = _try_extract_unsupported_param(last_error)
            if not param or param in removed or param not in payload:
                raise
            removed.add(param)
            payload.pop(param, None)
    raise RuntimeError(last_error or "LLM request failed")  # pragma: no cover


def _extract_text(payload):
    # OpenAI Responses API convenience field (when present)
    if "output_text" in payload and isinstance(payload.get("output_text"), str):
        text = payload.get("output_text") or ""
        if text.strip():
            return text

    if "output" in payload:
        out_list = payload.get("output") or []
        if out_list:
            # The first item is often "reasoning" (no user-visible text). Search all items.
            for out_item in out_list:
                if not isinstance(out_item, dict):
                    continue
                if isinstance(out_item.get("text"), str) and out_item.get("text").strip():
                    return out_item["text"]
                content = out_item.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, str) and part.strip():
                            return part
                        if isinstance(part, dict):
                            # Typical: {"type":"output_text","text":"..."}
                            if isinstance(part.get("text"), str) and part.get("text").strip():
                                return part["text"]
                            # Some variants nest text in "value"
                            if isinstance(part.get("value"), str) and part.get("value").strip():
                                return part["value"]
    if "choices" in payload:
        return payload["choices"][0]["message"]["content"]
    if "content" in payload and isinstance(payload["content"], list):
        return payload["content"][0].get("text", "")
    if "candidates" in payload:
        parts = payload["candidates"][0]["content"]["parts"]
        return parts[0].get("text", "")
    top_keys = ", ".join(sorted(payload.keys())) if isinstance(payload, dict) else type(payload).__name__
    raise RuntimeError(f"Unexpected LLM response format. Top-level keys: {top_keys}")


# Debug logging helpers

def _should_log(config):
    if os.getenv("LLM_DEBUG", "").lower() in ("1", "true", "yes", "on"):
        return True
    return bool(config.get("debug_log"))


def _log_target_file():
    path = os.getenv("LLM_DEBUG_FILE", "").strip()
    return Path(path) if path else None


def _log_target_files():
    seen = set()
    paths = []
    for key in ("LLM_DEBUG_FILE", "LLM_DEBUG_FILE_SECONDARY"):
        raw = os.getenv(key, "").strip()
        if not raw:
            continue
        p = Path(raw)
        ident = str(p)
        if ident in seen:
            continue
        seen.add(ident)
        paths.append(p)
    return paths


def _ensure_default_debug_file(provider):
    if os.getenv("LLM_DEBUG_FILE"):
        return
    if provider != "openai":
        return
    default_path = Path(__file__).resolve().parent / "logs" / "llm" / "openai_llm_debug.log"
    os.environ["LLM_DEBUG_FILE"] = str(default_path)


def _log_llm_call(provider, model, prompt, response):
    # Truncate to avoid huge logs
    max_len = 4000
    prompt_trunc = (prompt[:max_len] + "... [truncated]") if len(prompt) > max_len else prompt
    resp_trunc = (response[:max_len] + "... [truncated]") if len(response) > max_len else response
    lines = [
        f"[LLM DEBUG] provider={provider} model={model}",
        "PROMPT:",
        prompt_trunc,
        "RESPONSE:",
        resp_trunc,
        "-" * 40,
    ]
    out = "\n".join(lines)
    targets = _log_target_files()
    if targets:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as f:
                f.write(out + "\n")
    else:
        print(out, file=sys.stderr)


def _maybe_log_http_response(debug_meta, url, status, body):
    if not debug_meta:
        return
    config = debug_meta.get("config") or {}
    if not _should_log(config):
        return
    provider = debug_meta.get("provider", "unknown")
    model = debug_meta.get("model", "unknown")

    # Avoid huge logs; override via env var if needed.
    max_len = int(os.getenv("LLM_DEBUG_RAW_MAX", "20000"))
    body = body or ""
    body_trunc = (body[:max_len] + "... [truncated]") if len(body) > max_len else body

    status_str = str(status) if status is not None else "unknown"
    lines = [
        f"[LLM HTTP] provider={provider} model={model} status={status_str}",
        f"URL: {url}",
        "RESPONSE BODY:",
        body_trunc,
        "-" * 40,
    ]
    out = "\n".join(lines)
    targets = _log_target_files()
    if targets:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as f:
                f.write(out + "\n")
    else:
        print(out, file=sys.stderr)
