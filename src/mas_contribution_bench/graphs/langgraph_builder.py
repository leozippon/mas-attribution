"""Graph execution layer.

The class exposes a LangGraph-compatible boundary but also provides a pure
Python fallback used by tests and dry-runs. This keeps the benchmark runnable
before a real LLM backend is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mas_contribution_bench.agents.base import BaseAgent
from mas_contribution_bench.graphs.architectures import (
    downstream_roles,
    execution_order,
    fallback_final_answer_roles,
    upstream_roles,
)


@dataclass
class MASExecutionResult:
    state: dict[str, Any]
    final_answer: str


class MASGraphBuilder:
    def __init__(
        self,
        architecture,
        agents: dict[str, BaseAgent],
        removed_agents: set[str] | None = None,
        null_replacement: bool = False,
    ):
        self.architecture = architecture
        self.agents = agents
        self.removed_agents = removed_agents or set()
        self.null_replacement = null_replacement

    def build(self):
        return self

    def invoke(self, state: dict[str, Any]) -> MASExecutionResult:
        state = dict(state)
        state.setdefault("messages", [])
        state.setdefault("agent_outputs", {})
        state.setdefault("topology_messages", {})

        order = execution_order(self.architecture)

        for role in order:
            if role not in self.agents:
                continue

            role_state = self._state_for_role(state, role)

            if role in self.removed_agents and not self.null_replacement:
                continue

            if role in self.removed_agents and self.null_replacement:
                content = (
                    '{"summary": "null agent replacement", "artifact": "", '
                    '"evidence": [], "confidence": "low", "failure_modes": []}'
                )
                output = {
                    "agent_id": role,
                    "role": role,
                    "content": content,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "tool_calls": 0,
                    "metadata": {"null_agent": True},
                }
            else:
                agent_output = self.agents[role].invoke(role_state)
                output = {
                    "agent_id": agent_output.agent_id,
                    "role": agent_output.role,
                    "content": agent_output.content,
                    "input_tokens": agent_output.input_tokens,
                    "output_tokens": agent_output.output_tokens,
                    "tool_calls": agent_output.tool_calls,
                    "metadata": agent_output.metadata or {},
                }

            state["agent_outputs"][role] = output

            downstream = self._downstream(role)
            for dst in downstream:
                state["messages"].append(
                    {
                        "sender": role,
                        "receiver": dst,
                        "content": output["content"],
                    }
                )

            if not downstream:
                state["messages"].append(
                    {
                        "sender": role,
                        "receiver": "next",
                        "content": output["content"],
                    }
                )

        if self._fallback_enabled():
            final_answer = self._fallback_final_answer(state, order)
        else:
            final_answer = self._default_final_answer(state, order)

        state["final_answer"] = final_answer
        return MASExecutionResult(state=state, final_answer=final_answer)

    def _state_for_role(self, state: dict[str, Any], role: str) -> dict[str, Any]:
        role_state = dict(state)

        upstream = upstream_roles(self.architecture, role)
        if upstream:
            upstream_outputs = {
                src: state.get("agent_outputs", {}).get(src)
                for src in upstream
                if src in state.get("agent_outputs", {})
            }
        else:
            upstream_outputs = dict(state.get("agent_outputs", {}))

        role_state["upstream_roles"] = upstream
        role_state["upstream_outputs"] = upstream_outputs
        role_state["topology_context"] = {
            "role": role,
            "upstream_roles": upstream,
            "architecture_id": getattr(self.architecture, "architecture_id", None),
            "family": getattr(self.architecture, "family", None),
            "edges": getattr(self.architecture, "edges", []),
        }

        if upstream_outputs:
            role_state["messages"] = [
                {
                    "sender": src,
                    "receiver": role,
                    "content": output.get("content", ""),
                }
                for src, output in upstream_outputs.items()
                if output
            ]

        return role_state

    def _downstream(self, role: str) -> list[str]:
        return downstream_roles(self.architecture, role)

    def _fallback_enabled(self) -> bool:
        policy = getattr(self.architecture, "orchestration", {}).get("fallback_final_answer", {})
        return bool(policy.get("enabled", False))

    def _is_null_output(self, output: dict[str, Any] | None) -> bool:
        if not output:
            return True
        if output.get("metadata", {}).get("null_agent"):
            return True
        content = output.get("content") or ""
        return not str(content).strip()

    def _has_final_answer_permission(self, output: dict[str, Any] | None) -> bool:
        if not output:
            return False
        permissions = (output.get("metadata") or {}).get("permissions") or {}
        return bool(permissions.get("final_answer", False))

    def _respect_final_answer_permission(self, state: dict[str, Any]) -> bool:
        return bool(state.get("respect_final_answer_permission", False))

    def _record_final_answer_choice(
        self,
        state: dict[str, Any],
        *,
        source_role: str | None,
        fallback_used: bool,
        no_authorized_final_answer: bool = False,
        unauthorized_candidates: list[str] | None = None,
    ) -> None:
        diagnostics = dict(state.get("permission_diagnostics") or {})
        diagnostics.update(
            {
                "final_answer_source_role": source_role,
                "fallback_used": fallback_used,
                "no_authorized_final_answer": no_authorized_final_answer,
                "unauthorized_final_answer_candidates": unauthorized_candidates or [],
            }
        )
        state["permission_diagnostics"] = diagnostics

    def _strict_final_answer_permission(self, state: dict[str, Any]) -> bool:
        return bool(
            self._respect_final_answer_permission(state)
            and state.get("strict_permission_enforcement", False)
        )

    def _default_final_answer(self, state: dict[str, Any], order: list[str]) -> str:
        final_role = self._final_role(order)
        terminal_output = state.get("agent_outputs", {}).get(final_role) if final_role else None
        if not isinstance(terminal_output, dict) or self._is_null_output(terminal_output):
            self._record_final_answer_choice(
                state,
                source_role=final_role,
                fallback_used=False,
            )
            return ""
        output = terminal_output
        if not self._respect_final_answer_permission(state) or self._has_final_answer_permission(output):
            self._record_final_answer_choice(
                state,
                source_role=final_role,
                fallback_used=False,
            )
            return str(output.get("content", ""))

        unauthorized_candidates = []
        if self._respect_final_answer_permission(state):
            for role in reversed(order):
                candidate = state.get("agent_outputs", {}).get(role)
                if not isinstance(candidate, dict) or self._is_null_output(candidate):
                    continue
                if self._has_final_answer_permission(candidate):
                    self._record_final_answer_choice(
                        state,
                        source_role=role,
                        fallback_used=True,
                    )
                    return str(candidate.get("content", ""))
                unauthorized_candidates.append(role)

        if self._strict_final_answer_permission(state):
            self._record_final_answer_choice(
                state,
                source_role=None,
                fallback_used=True,
                no_authorized_final_answer=True,
                unauthorized_candidates=unauthorized_candidates,
            )
            return ""

        if state["agent_outputs"]:
            source_role = next(reversed(state["agent_outputs"]))
            self._record_final_answer_choice(
                state,
                source_role=source_role,
                fallback_used=True,
                unauthorized_candidates=unauthorized_candidates,
            )
            return str(list(state["agent_outputs"].values())[-1].get("content", ""))

        self._record_final_answer_choice(
            state,
            source_role=None,
            fallback_used=False,
            no_authorized_final_answer=True,
        )
        return ""

    def _fallback_final_answer(self, state: dict[str, Any], order: list[str]) -> str:
        policy = getattr(self.architecture, "orchestration", {}).get("fallback_final_answer", {})
        role_priority = policy.get("role_priority") or [
            "coder",
            "executor",
            "verifier",
            "tester",
            "critic",
            "reviewer",
            "debugger",
            "planner",
            "researcher",
            "retriever",
            "supervisor",
            "memory_manager",
            "tool_agent",
            "finalizer",
            "aggregator",
        ]

        candidates = fallback_final_answer_roles(self.architecture, role_priority=role_priority)

        authorized_candidates = []
        fallback_candidates = []
        for role in candidates:
            output = state.get("agent_outputs", {}).get(role)
            if self._is_null_output(output):
                continue
            if self._respect_final_answer_permission(state):
                if self._has_final_answer_permission(output):
                    authorized_candidates.append(role)
                else:
                    fallback_candidates.append(role)
                continue
            self._record_final_answer_choice(
                state,
                source_role=role,
                fallback_used=False,
            )
            return str(output.get("content", ""))

        if authorized_candidates:
            output = state.get("agent_outputs", {}).get(authorized_candidates[0])
            self._record_final_answer_choice(
                state,
                source_role=authorized_candidates[0],
                fallback_used=authorized_candidates[0] not in self._terminal_sources(),
                unauthorized_candidates=fallback_candidates,
            )
            return str(output.get("content", "")) if output else ""

        remaining = list(reversed(order))
        remaining_unauthorized = []
        if self._respect_final_answer_permission(state):
            for role in remaining:
                output = state.get("agent_outputs", {}).get(role)
                if self._is_null_output(output):
                    continue
                if self._has_final_answer_permission(output):
                    self._record_final_answer_choice(
                        state,
                        source_role=role,
                        fallback_used=True,
                        unauthorized_candidates=fallback_candidates,
                    )
                    return str(output.get("content", ""))
                remaining_unauthorized.append(role)

        if self._strict_final_answer_permission(state):
            self._record_final_answer_choice(
                state,
                source_role=None,
                fallback_used=True,
                no_authorized_final_answer=True,
                unauthorized_candidates=fallback_candidates + remaining_unauthorized,
            )
            return ""

        for role in fallback_candidates + remaining:
            output = state.get("agent_outputs", {}).get(role)
            if self._is_null_output(output):
                continue
            self._record_final_answer_choice(
                state,
                source_role=role,
                fallback_used=True,
                unauthorized_candidates=fallback_candidates,
            )
            return str(output.get("content", ""))

        self._record_final_answer_choice(
            state,
            source_role=None,
            fallback_used=True,
            no_authorized_final_answer=True,
            unauthorized_candidates=fallback_candidates,
        )
        return ""

    def _terminal_sources(self) -> list[str]:
        return [
            src
            for src, dst in getattr(self.architecture, "edges", [])
            if dst == "final_answer"
        ]

    def _final_role(self, order: list[str]) -> str | None:
        terminal_sources = []
        for src, dst in getattr(self.architecture, "edges", []):
            if dst == "final_answer":
                terminal_sources.append(src)

        for role in reversed(order):
            if role in terminal_sources:
                return role

        for preferred in ("finalizer", "aggregator", "supervisor", "verifier", "coder", "executor"):
            if preferred in order:
                return preferred

        return order[-1] if order else None
