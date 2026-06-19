from __future__ import annotations

import json
import http.client
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable


class LLMConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int | None = None,
    ) -> None:
        self._load_dotenv()
        self.api_key = (
            api_key
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        self.model = (
            model
            or os.getenv("DEEPSEEK_MODEL")
            or os.getenv("OPENAI_MODEL")
            or os.getenv("LLM_MODEL")
            or "deepseek-chat"
        )
        self.base_url = (
            base_url
            or os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("LLM_MAX_RETRIES", "3"))
        if not self.api_key:
            raise LLMConfigurationError("Missing DEEPSEEK_API_KEY, OPENAI_API_KEY, or LLM_API_KEY")
        if not self.model:
            raise LLMConfigurationError("Missing OPENAI_MODEL or LLM_MODEL")

    @staticmethod
    def _load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
        env_path = Path(path)
        if not env_path.exists():
            return
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

    def chat_with_tools_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolSpec],
        max_iterations: int = 12,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tool_map = {tool.name: tool for tool in tools}
        trace: list[dict[str, Any]] = []

        for _ in range(max_iterations):
            response = self._chat(messages, [tool.to_openai_tool() for tool in tools])
            message = response["choices"][0]["message"]
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return messages, trace

            for tool_call in tool_calls:
                function = tool_call["function"]
                name = function["name"]
                arguments = json.loads(function.get("arguments") or "{}")
                if name not in tool_map:
                    result = {"valid": False, "error": f"Unknown tool: {name}"}
                else:
                    result = tool_map[name].handler(**arguments)
                trace.append(
                    {
                        "tool_call_id": tool_call.get("id"),
                        "tool": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return messages, trace

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.4,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.max_retries) + 1):
            try:
                return self._chat_once(payload)
            except urllib.error.HTTPError as exc:
                body = self._read_error_body(exc)
                if not self._is_retriable_http(exc.code) or attempt >= self.max_retries:
                    raise RuntimeError(f"LLM HTTP {exc.code}: {body}") from exc
                last_error = RuntimeError(f"LLM HTTP {exc.code}: {body}")
            except self._transient_errors() as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"LLM transport failed after {self.max_retries} attempts: {type(exc).__name__}: {exc}"
                    ) from exc
                last_error = exc
            self._sleep_before_retry(attempt)
        raise RuntimeError(f"LLM request failed: {last_error}")

    def _chat_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _is_retriable_http(code: int) -> bool:
        return code == 429 or 500 <= code <= 599

    @staticmethod
    def _transient_errors() -> tuple[type[BaseException], ...]:
        return (
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            TimeoutError,
            ConnectionResetError,
            socket.timeout,
            urllib.error.URLError,
            JSONDecodeError,
        )

    @staticmethod
    def _read_error_body(exc: urllib.error.HTTPError) -> str:
        try:
            return exc.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _sleep_before_retry(attempt: int) -> None:
        time.sleep(min(2 ** (attempt - 1), 8))
