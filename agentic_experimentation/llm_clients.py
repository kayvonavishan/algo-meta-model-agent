import json
import os
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
        self.config = config
        self.api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"), "")
        self.base_url = config.get("base_url") or "https://api.openai.com/v1"

    def generate(self, prompt, system_prompt=None):
        if not self.api_key:
            raise RuntimeError("OpenAI API key env var not set.")
        url = f"{self.base_url}/responses"
        inputs = []
        if system_prompt:
            inputs.append({"role": "system", "content": system_prompt})
        inputs.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.get("model", "gpt-4.1"),
            "input": inputs,
            "temperature": self.config.get("temperature", 0.2),
            "max_output_tokens": self.config.get("max_tokens", 1200),
        }
        reasoning = self.config.get("reasoning")
        if reasoning:
            payload["reasoning"] = {"effort": reasoning}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        result = _post_json(url, payload, headers, self.config.get("timeout_sec", 60))
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
        result = _post_json(url, payload, headers, self.config.get("timeout_sec", 60))
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
        result = _post_json(url, payload, headers, self.config.get("timeout_sec", 60))
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


def _post_json(url, payload, headers, timeout_sec):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM request failed: {exc.code} {detail}") from exc

    parsed = json.loads(raw)
    return _extract_text(parsed)


def _extract_text(payload):
    if "output" in payload:
        out_list = payload.get("output") or []
        if out_list:
            content = out_list[0].get("content") if isinstance(out_list[0], dict) else None
            if isinstance(content, list) and content:
                text_part = content[0]
                if isinstance(text_part, dict):
                    return text_part.get("text", "")
                if isinstance(text_part, str):
                    return text_part
    if "choices" in payload:
        return payload["choices"][0]["message"]["content"]
    if "content" in payload and isinstance(payload["content"], list):
        return payload["content"][0].get("text", "")
    if "candidates" in payload:
        parts = payload["candidates"][0]["content"]["parts"]
        return parts[0].get("text", "")
    raise RuntimeError("Unexpected LLM response format.")


# Debug logging helpers

def _should_log(config):
    if os.getenv("LLM_DEBUG", "").lower() in ("1", "true", "yes", "on"):
        return True
    return bool(config.get("debug_log"))


def _log_target_file():
    path = os.getenv("LLM_DEBUG_FILE", "").strip()
    return Path(path) if path else None


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
    target = _log_target_file()
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(out + "\n")
    else:
        print(out, file=sys.stderr)
