"""Agent abstractions used by the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, ClassVar, Protocol

import requests

from mas_contribution_bench.data.external import public_task_view


class ModelClient(Protocol):
    """Minimal model interface.

    A real implementation can wrap LangChain/OpenAI/etc. The benchmark core only
    requires a deterministic `complete` method returning text.
    """

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        ...


@dataclass
class AgentOutput:
    agent_id: str
    role: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    metadata: dict[str, Any] | None = None


class DryRunModelClient:
    """Offline model client for smoke tests and framework validation."""

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        role = kwargs.get("role", "agent")
        task_id = kwargs.get("task_id", "unknown_task")
        last = messages[-1]["content"] if messages else ""
        return (
            "{\n"
            f'  "summary": "Dry-run {role} response for {task_id}.",\n'
            f'  "artifact": {last[:500]!r},\n'
            '  "evidence": [],\n'
            '  "confidence": "low",\n'
            '  "failure_modes": []\n'
            "}"
        )


class DeepSeekModelClient:
    """OpenAI-compatible chat-completions client, defaulting to DeepSeek.

    The API key is read from the configured environment variable at call time.
    Keep keys in environment variables or a local .env file that is never committed.
    """

    _shared_cache_indexes: ClassVar[dict[str, dict[str, dict[str, Any]]]] = {}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        provider_name: str = "deepseek",
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url_env: str = "DEEPSEEK_BASE_URL",
        model_env: str = "DEEPSEEK_MODEL",
        default_base_url: str = "https://api.deepseek.com",
        default_model_name: str = "deepseek-chat",
    ):
        self.provider_name = provider_name
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.model_env = model_env
        self.api_key = api_key
        self.base_url = (base_url or os.getenv(base_url_env) or default_base_url).rstrip("/")
        self.default_model = default_model or os.getenv(model_env) or default_model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = int(os.getenv(f"{provider_name.upper()}_MAX_RETRIES", str(max_retries if max_retries is not None else 3)))
        self.retry_backoff_seconds = float(os.getenv(f"{provider_name.upper()}_RETRY_BACKOFF", str(retry_backoff_seconds if retry_backoff_seconds is not None else 2.0)))
        self.last_usage: dict[str, Any] = {}
        self.last_cache_metadata: dict[str, Any] = {}

    def _cache_enabled(self) -> bool:
        return os.getenv("MAS_LLM_CACHE_ENABLED", "1").lower() not in {"0", "false", "no", "n"}

    def _prepare_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [dict(message) for message in messages]

    def _cache_path(self, model: str) -> Path:
        configured = os.getenv("MAS_LLM_CACHE_FILE")
        if configured:
            return Path(configured)
        cache_dir = Path(os.getenv("MAS_LLM_CACHE_DIR", "data/cache/llm_calls"))
        safe_provider = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in self.provider_name)
        safe_model = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model)
        return cache_dir / f"{safe_provider}_{safe_model}.jsonl"

    def _cache_key(self, payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _load_cache_index(self, path: Path) -> dict[str, dict[str, Any]]:
        cache_id = str(path)
        if cache_id in self._shared_cache_indexes:
            return self._shared_cache_indexes[cache_id]
        index: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = row.get("cache_key")
                    if key:
                        index[str(key)] = row
        self._shared_cache_indexes[cache_id] = index
        return index

    def _append_cache_row(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _llm_error_fallback_enabled(self) -> bool:
        env = os.getenv("MAS_LLM_ERROR_FALLBACK")
        if env is None:
            return False
        return env.lower() in {"1", "true", "yes", "y"}

    def _fallback_content(
        self,
        *,
        error_type: str,
        error_message: str,
        model: str,
        max_tokens: int,
        cache_key: str,
        cache_path: Path,
    ) -> str:
        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_error": True,
            "llm_error_type": error_type,
        }
        self.last_cache_metadata = {
            "local_cache_hit": False,
            "local_cache_key": cache_key,
            "local_cache_path": str(cache_path),
            "provider_cache_hit_tokens": 0,
            "provider_cache_miss_tokens": 0,
            "llm_error": True,
            "llm_error_type": error_type,
        }
        return json.dumps(
            {
                "summary": f"{self.provider_name} request failed and was converted to a benchmark failure record.",
                "artifact": "",
                "evidence": [],
                "confidence": "low",
                "failure_modes": ["llm_request_failed", error_type],
                "llm_error": {
                    "provider": self.provider_name,
                    "model": model,
                    "max_tokens": max_tokens,
                    "error_type": error_type,
                    "error_message": error_message[:1000],
                },
            },
            ensure_ascii=False,
        )

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.last_usage = {}
        self.last_cache_metadata = {}
        model = kwargs.get("model") or self.default_model
        if isinstance(model, str) and model.startswith("${"):
            model = self.default_model
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 2048)

        request_messages = self._prepare_messages(messages)
        payload = {
            "model": model,
            "messages": request_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        seed = kwargs.get("seed")
        if seed is not None:
            payload["seed"] = int(seed)
        cache_path = self._cache_path(str(model))
        cache_key = self._cache_key(payload)
        if self._cache_enabled():
            cache_index = self._load_cache_index(cache_path)
            cached = cache_index.get(cache_key)
            if cached is not None:
                self.last_usage = dict(cached.get("usage") or {})
                self.last_cache_metadata = {
                    "local_cache_hit": True,
                    "local_cache_key": cache_key,
                    "local_cache_path": str(cache_path),
                    "provider_cache_hit_tokens": self.last_usage.get("prompt_cache_hit_tokens", 0),
                    "provider_cache_miss_tokens": self.last_usage.get("prompt_cache_miss_tokens", 0),
                }
                return str(cached.get("content", ""))

        api_key = self.api_key or os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} is not set. Export it before using MAS_MODEL_BACKEND={self.provider_name}."
            )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code >= 500 and attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (2 ** attempt))
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(f"{self.provider_name} API error {response.status_code}: {response.text[:1000]}")
                data = response.json()
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    if self._llm_error_fallback_enabled():
                        return self._fallback_content(
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            model=str(model),
                            max_tokens=int(max_tokens),
                            cache_key=cache_key,
                            cache_path=cache_path,
                        )
                    raise RuntimeError(f"{self.provider_name} API request failed after {self.max_retries + 1} attempts: {exc}") from exc
                time.sleep(self.retry_backoff_seconds * (2 ** attempt))
        else:
            if self._llm_error_fallback_enabled():
                return self._fallback_content(
                    error_type=type(last_error).__name__ if last_error else "UnknownError",
                    error_message=str(last_error),
                    model=str(model),
                    max_tokens=int(max_tokens),
                    cache_key=cache_key,
                    cache_path=cache_path,
                )
            raise RuntimeError(f"{self.provider_name} API request failed: {last_error}")
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected {self.provider_name} response: {data}") from exc
        self.last_usage = dict(data.get("usage") or {})
        self.last_cache_metadata = {
            "local_cache_hit": False,
            "local_cache_key": cache_key,
            "local_cache_path": str(cache_path),
            "provider_cache_hit_tokens": self.last_usage.get("prompt_cache_hit_tokens", 0),
            "provider_cache_miss_tokens": self.last_usage.get("prompt_cache_miss_tokens", 0),
        }
        if self._cache_enabled():
            row = {
                "cache_key": cache_key,
                "created_at": int(time.time()),
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": request_messages,
                "content": content,
                "usage": self.last_usage,
            }
            self._append_cache_row(cache_path, row)
            self._load_cache_index(cache_path)[cache_key] = row
        return content


class OpenAICompatibleModelClient(DeepSeekModelClient):
    """Generic OpenAI-compatible client for local vLLM/SGLang/Ollama-style servers."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        provider_name: str = "vllm",
    ):
        self.disable_thinking = os.getenv("QWEN_DISABLE_THINKING", "1").lower() not in {"0", "false", "no", "n"}
        super().__init__(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or "dummy",
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("VLLM_BASE_URL") or "http://127.0.0.1:8000/v1",
            default_model=(
                default_model
                or os.getenv("MODEL_NAME")
                or os.getenv("OPENAI_MODEL")
                or os.getenv("VLLM_MODEL")
                or "qwen-local"
            ),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            provider_name=provider_name,
            api_key_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL",
            model_env="MODEL_NAME",
            default_base_url="http://127.0.0.1:8000/v1",
            default_model_name="qwen-local",
        )

    def _prepare_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        prepared = [dict(message) for message in messages]
        if not self.disable_thinking:
            return prepared
        for message in reversed(prepared):
            if message.get("role") == "user":
                content = str(message.get("content", ""))
                if "/no_think" not in content:
                    message["content"] = content.rstrip() + "\n\n/no_think"
                break
        return prepared


class BaseAgent:
    def __init__(
        self,
        agent_id: str,
        role: str,
        prompt: str,
        permissions: dict[str, bool],
        model_client: ModelClient | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.prompt = prompt
        self.permissions = permissions
        self.model_client = model_client or DryRunModelClient()
        self.model_kwargs = model_kwargs or {}

    def _dataset_output_instruction(self, dataset: str) -> str:
        if dataset in {"swebench_lite", "swebench_verified", "teambench"}:
            return (
                "Official evaluation output requirement:\n"
                "Return the final artifact as a valid unified git diff patch starting with 'diff --git a/...'. "
                "Do not return prose instead of a patch. If no file change is needed, return an empty patch artifact."
            )
        if dataset == "livecodebench":
            return (
                "Official evaluation output requirement:\n"
                "Return only executable Python solution code in the final artifact. Do not include explanations."
            )
        if dataset == "arc_agi_2":
            return (
                "Official evaluation output requirement:\n"
                "Return only the predicted output grid as JSON, e.g. [[1,2],[3,4]]."
            )
        if dataset in {"aime_2026", "gpqa_diamond", "hle"}:
            return (
                "Official evaluation output requirement:\n"
                "Return the final answer exactly and concisely in the final artifact. Avoid extra explanation in the artifact."
            )
        return ""

    def build_messages(self, state: dict[str, Any]) -> list[dict[str, str]]:
        task = public_task_view(state.get("task", {}) or {})
        history = state.get("messages", [])
        history_text = "\n".join(
            f"{item.get('sender')}: {item.get('content')}" for item in history[-8:]
        )
        dataset = str(task.get("dataset") or "").lower()
        task_details = [
            f"Task ID: {task.get('task_id')}",
            f"Dataset: {task.get('dataset')}",
            f"Prompt:\n{task.get('prompt', '')}",
        ]
        if task.get("context"):
            task_details.append(f"Context:\n{task.get('context')}")
        if task.get("output_format"):
            task_details.append(f"Output format: {task.get('output_format')}")
        output_instruction = self._dataset_output_instruction(dataset)
        if output_instruction:
            task_details.append(output_instruction)
        if task.get("entry_point"):
            task_details.append(f"Required entry point: {task.get('entry_point')}")
        permission_lines = [
            f"- {name}: {str(value).lower()}"
            for name, value in sorted(self.permissions.items())
        ]
        task_details.append(
            "Current role permissions:\n"
            + "\n".join(permission_lines)
            + "\nOnly perform actions and claim authority that are enabled above."
        )
        user_content = (
            "\n\n".join(task_details)
            + f"\n\nRecent collaboration history:\n{history_text}"
        )
        return [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": user_content},
        ]

    def _explicit_permission_overrides(self, state: dict[str, Any]) -> dict[str, bool]:
        overrides = state.get("permission_overrides") or {}
        role_overrides = overrides.get(self.agent_id) or {}
        return {str(name): bool(value) for name, value in role_overrides.items()}

    def _enforce_output_permissions(
        self,
        content: str,
        state: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Apply runtime permission constraints to an agent response.

        Permission sets are normally role affordances. For intervention
        experiments, only explicitly overridden permissions are treated as hard
        causal interventions so normal aggregator/finalizer behavior is not
        accidentally disabled by its base permission profile.
        """

        diagnostics: dict[str, Any] = {
            "permission_enforced": bool(state.get("strict_permission_enforcement", False)),
            "blocked_outputs": [],
            "blocked_tool_calls": [],
            "explicit_permission_overrides": self._explicit_permission_overrides(state),
        }
        if not diagnostics["permission_enforced"]:
            return content, diagnostics

        explicit = diagnostics["explicit_permission_overrides"]
        if explicit.get("call_tools") is False:
            diagnostics["blocked_tool_calls"].append("all_tools")
        if explicit.get("run_tests") is False:
            diagnostics["blocked_tool_calls"].append("run_tests")

        if explicit.get("write_solution") is not False:
            return content, diagnostics

        diagnostics["blocked_outputs"].append("solution_artifact")
        replacement = {
            "summary": (
                f"{self.agent_id} output was blocked by runtime permission "
                "enforcement because write_solution=false."
            ),
            "artifact": "",
            "evidence": [],
            "confidence": "low",
            "failure_modes": ["permission_blocked_write_solution"],
            "permission_blocked": True,
            "blocked_permission": "write_solution",
        }
        return json.dumps(replacement, ensure_ascii=False), diagnostics

    def invoke(self, state: dict[str, Any]) -> AgentOutput:
        messages = self.build_messages(state)
        task = state.get("task") or {}
        dataset = str(task.get("dataset") or "").lower()
        model_kwargs = dict(self.model_kwargs)
        if dataset == "arc_agi_2":
            # ARC-AGI-2 grids are long but the required answer is only a JSON grid.
            # Keeping completion short prevents vLLM context overflow on 16k servers.
            current_max_tokens = model_kwargs.get("max_tokens")
            model_kwargs["max_tokens"] = min(int(current_max_tokens or 256), 256)
        if model_kwargs.get("seed") is None and state.get("seed") is not None:
            model_kwargs["seed"] = int(state["seed"])
        content = self.model_client.complete(
            messages,
            role=self.role,
            agent_id=self.agent_id,
            task_id=task.get("task_id"),
            **model_kwargs,
        )
        content, permission_diagnostics = self._enforce_output_permissions(content, state)
        usage = dict(getattr(self.model_client, "last_usage", {}) or {})
        cache_metadata = dict(getattr(self.model_client, "last_cache_metadata", {}) or {})
        input_tokens = sum(len(m["content"].split()) for m in messages)
        output_tokens = len(content.split())
        if usage:
            input_tokens = int(usage.get("prompt_tokens") or input_tokens)
            output_tokens = int(usage.get("completion_tokens") or output_tokens)
        return AgentOutput(
            agent_id=self.agent_id,
            role=self.role,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "permissions": self.permissions,
                "permission_diagnostics": permission_diagnostics,
                "model_usage": usage,
                "cache": cache_metadata,
            },
        )
