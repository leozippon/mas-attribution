"""Architecture helpers."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import networkx as nx

from mas_contribution_bench.config.loaders import LoadedArchitectureSpec
from mas_contribution_bench.data.schemas import GraphSpec


def architecture_to_graph_spec(architecture: LoadedArchitectureSpec) -> GraphSpec:
    nodes = sorted(set(architecture.roles) | {"final_answer"})
    return GraphSpec(
        nodes=nodes,
        edges=architecture.edges,
        edge_type="communication",
        graph_metadata={
            "architecture_id": architecture.architecture_id,
            "family": architecture.family,
            "orchestration": architecture.orchestration,
        },
    )


def architecture_to_networkx(architecture: LoadedArchitectureSpec) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(architecture.roles)
    graph.add_node("final_answer")
    graph.add_edges_from(architecture.edges)
    return graph


def role_edges(architecture: LoadedArchitectureSpec) -> list[tuple[str, str]]:
    roles = set(architecture.roles)
    return [
        (src, dst)
        for src, dst in architecture.edges
        if src in roles and dst in roles
    ]


def upstream_roles(architecture: LoadedArchitectureSpec, role: str) -> list[str]:
    roles = set(architecture.roles)
    return [
        src
        for src, dst in architecture.edges
        if dst == role and src in roles
    ]


def downstream_roles(architecture: LoadedArchitectureSpec, role: str) -> list[str]:
    roles = set(architecture.roles)
    return [
        dst
        for src, dst in architecture.edges
        if src == role and dst in roles
    ]

def terminal_source_roles(architecture: LoadedArchitectureSpec) -> list[str]:
    roles = set(architecture.roles)
    return [
        src
        for src, dst in architecture.edges
        if dst == "final_answer" and src in roles
    ]


def fallback_final_answer_roles(
    architecture: LoadedArchitectureSpec,
    role_priority: list[str] | None = None,
) -> list[str]:
    """Rank fallback final-answer roles by topology distance, then role priority.

    The search starts from nodes that point to final_answer. If the terminal
    node itself is unavailable, its nearest upstream role is preferred. Ties are
    broken by solution-bearing role priority.
    """

    role_priority = role_priority or [
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
    priority = {role: idx for idx, role in enumerate(role_priority)}

    roles = set(architecture.roles)
    reverse_edges: dict[str, list[str]] = defaultdict(list)
    for src, dst in role_edges(architecture):
        reverse_edges[dst].append(src)

    starts = terminal_source_roles(architecture)
    if not starts:
        starts = list(architecture.roles)

    distances: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque((role, 0) for role in starts if role in roles)

    while queue:
        role, dist = queue.popleft()
        if role in distances:
            continue
        distances[role] = dist
        for parent in reverse_edges.get(role, []):
            if parent in roles and parent not in distances:
                queue.append((parent, dist + 1))

    for role in architecture.roles:
        distances.setdefault(role, 10_000)

    return sorted(
        architecture.roles,
        key=lambda role: (
            distances.get(role, 10_000),
            priority.get(role, len(priority)),
            role,
        ),
    )


def _topological_or_role_order(architecture: LoadedArchitectureSpec) -> list[str]:
    graph = nx.DiGraph()
    graph.add_nodes_from(architecture.roles)
    graph.add_edges_from(role_edges(architecture))

    if graph.number_of_edges() > 0 and nx.is_directed_acyclic_graph(graph):
        return list(nx.topological_sort(graph))

    return list(architecture.roles)


def _debate_order(architecture: LoadedArchitectureSpec) -> list[str]:
    """Return a deterministic order for cyclic/debate graphs.

    LangGraph can model cycles, but the current benchmark runner executes a
    bounded single-pass fallback. For cyclic controlled debate variants, we use
    the declared role order and move finalizer/aggregator-like roles to the end.
    """

    roles = list(architecture.roles)
    terminal_like = [
        role for role in roles
        if role in {"finalizer", "aggregator", "supervisor"}
    ]
    non_terminal = [role for role in roles if role not in set(terminal_like)]
    return non_terminal + terminal_like


def execution_order(architecture: LoadedArchitectureSpec) -> list[str]:
    """Return the role execution order implied by the architecture.

    Controlled topology experiments must respect edges even when
    orchestration.execution_mode is absent. DAG-like graphs are topologically
    sorted; cyclic debate/star graphs fall back to a deterministic single-pass
    order with finalizer-like roles last.
    """

    graph = nx.DiGraph()
    graph.add_nodes_from(architecture.roles)
    graph.add_edges_from(role_edges(architecture))

    if graph.number_of_edges() > 0 and nx.is_directed_acyclic_graph(graph):
        return list(nx.topological_sort(graph))

    mode = architecture.orchestration.get("execution_mode", "")
    if mode in {"debate", "cyclic", "round_robin"}:
        return _debate_order(architecture)

    return _debate_order(architecture)


def bypass_edges(edges: list[tuple[str, str]], removed: set[str]) -> list[tuple[str, str]]:
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    kept: list[tuple[str, str]] = []

    for src, dst in edges:
        if dst in removed:
            incoming[dst].append(src)
        if src in removed:
            outgoing[src].append(dst)
        if src not in removed and dst not in removed:
            kept.append((src, dst))

    for node in removed:
        for src in incoming[node]:
            for dst in outgoing[node]:
                if src not in removed and dst not in removed and src != dst:
                    kept.append((src, dst))

    return sorted(set(kept))


def topology_features(architecture: LoadedArchitectureSpec) -> dict[str, dict[str, float]]:
    graph = architecture_to_networkx(architecture)
    roles = architecture.roles
    undirected = graph.to_undirected()

    degree = dict(graph.degree())
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    betweenness = nx.betweenness_centrality(graph) if graph.number_of_nodes() else {}
    closeness = nx.closeness_centrality(graph) if graph.number_of_nodes() else {}
    pagerank = nx.pagerank(graph) if graph.number_of_edges() else {n: 0.0 for n in graph.nodes}
    articulation = set(nx.articulation_points(undirected)) if undirected.number_of_nodes() else set()

    depths: dict[str, float] = {role: 0.0 for role in roles}
    if nx.is_directed_acyclic_graph(graph):
        for role in roles:
            ancestors = nx.ancestors(graph, role)
            depths[role] = float(
                max((len(nx.shortest_path(graph, a, role)) - 1 for a in ancestors), default=0)
            )

    return {
        role: {
            "degree": float(degree.get(role, 0)),
            "in_degree": float(in_degree.get(role, 0)),
            "out_degree": float(out_degree.get(role, 0)),
            "betweenness": float(betweenness.get(role, 0.0)),
            "closeness": float(closeness.get(role, 0.0)),
            "pagerank": float(pagerank.get(role, 0.0)),
            "dag_depth": float(depths.get(role, 0.0)),
            "fan_in": float(in_degree.get(role, 0)),
            "fan_out": float(out_degree.get(role, 0)),
            "is_articulation_point": float(role in articulation),
        }
        for role in roles
    }


def controlled_architecture(raw: dict[str, Any], architecture_id: str = "controlled") -> LoadedArchitectureSpec:
    roles = list(raw.get("roles") or raw.get("controlled_role_set") or [])
    edges = [tuple(edge) for edge in raw.get("edges", [])]

    orchestration = dict(raw.get("orchestration", {}))
    template = raw.get("template", "controlled")
    if "execution_mode" not in orchestration:
        if template in {"chain", "dag"}:
            orchestration["execution_mode"] = "dag"
        else:
            orchestration["execution_mode"] = "single_pass"
    orchestration.setdefault("max_rounds", 1)

    return LoadedArchitectureSpec(
        architecture_id=architecture_id,
        name=raw.get("name", architecture_id),
        family=raw.get("family", template),
        roles=roles,
        canonical_roles={role: role for role in roles},
        entrypoint=raw.get("entrypoint", roles[0] if roles else ""),
        terminal_nodes=list(raw.get("terminal_nodes", ["final_answer"])),
        edges=edges,
        orchestration=orchestration,
        default_permissions=dict(raw.get("default_permissions", {})),
        raw=raw,
    )