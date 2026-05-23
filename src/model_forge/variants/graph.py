from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from model_forge.variants.manifest import load_family, transform_from_variant, variant_config


def variant_graph(family: str) -> dict[str, Any]:
    family_config = load_family(family)
    variants = family_config.get("variants") or {}
    nodes = []
    edges = []
    children: dict[str, list[str]] = defaultdict(list)
    for variant_name, raw_variant in sorted(variants.items()):
        variant = variant_config(family_config, variant_name)
        source = variant.get("base_variant")
        transform = transform_from_variant(variant_name, variant)
        nodes.append(
            {
                "variant": variant_name,
                "source_variant": source,
                "repo_id": variant.get("repo_id"),
                "served_model_name": variant.get("served_model_name"),
                "transform_type": transform.get("type"),
                "objective": transform.get("objective"),
            }
        )
        if source:
            edges.append(
                {
                    "source": source,
                    "target": variant_name,
                    "transform": transform,
                }
            )
            children[str(source)].append(variant_name)
    return {
        "family": family,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "children": {key: sorted(value) for key, value in sorted(children.items())},
    }


def ancestry(graph: Mapping[str, Any], variant: str) -> list[str]:
    parent_by_variant = {edge["target"]: edge["source"] for edge in graph.get("edges", [])}
    path = [variant]
    current = variant
    while current in parent_by_variant:
        current = parent_by_variant[current]
        path.append(current)
    return list(reversed(path))
