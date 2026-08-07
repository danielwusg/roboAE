from __future__ import annotations

import ast
from pathlib import Path


TOOL_CAPABILITIES = ("detection", "grasp", "language", "pointing", "segmentation", "vision")
HEAVY_CAPABILITY = "language"

_METHOD_CAPABILITY = {
    "detect": "detection",
    "grasp": "grasp",
    "language": "language",
    "point": "pointing",
    "segment": "segmentation",
    "vision": "vision",
}

_TYPE_CAPABILITY = {
    "DetectionRequest": "detection",
    "DetectionResult": "detection",
    "Detection": "detection",
    "GraspRequest": "grasp",
    "GraspResult": "grasp",
    "GraspCandidate": "grasp",
    "LanguageRequest": "language",
    "PointingRequest": "pointing",
    "PointingResult": "pointing",
    "SegmentationRequest": "segmentation",
    "SegmentationResult": "segmentation",
    "VisionRequest": "vision",
}

_UNKNOWN_CALLS = frozenset({"getattr", "eval", "exec", "globals", "vars", "__import__"})


def _config_literal_nodes(tree: ast.AST) -> set[int]:
    marked: set[int] = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if not any(isinstance(item, ast.Name) and item.id == "SCAFFOLD_CONFIG" for item in targets):
            continue
        value = getattr(node, "value", None)
        if value is None:
            continue
        for child in ast.walk(value):
            marked.add(id(child))
    return marked


def declared_capabilities(source: str) -> frozenset[str] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else []
        if not any(isinstance(item, ast.Name) and item.id == "SCAFFOLD_CONFIG" for item in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(value, dict):
            return None
        names: set[str] = set()
        for key in ("required_capabilities", "optional_capabilities"):
            entry = value.get(key, ())
            if not isinstance(entry, (list, tuple)):
                return None
            names.update(str(item) for item in entry)
        return frozenset(names)
    return None


def required_capabilities(source: str) -> frozenset[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset(TOOL_CAPABILITIES)
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else []
        if not any(isinstance(item, ast.Name) and item.id == "SCAFFOLD_CONFIG" for item in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return frozenset(TOOL_CAPABILITIES)
        entry = value.get("required_capabilities", ()) if isinstance(value, dict) else ()
        if not isinstance(entry, (list, tuple)):
            return frozenset(TOOL_CAPABILITIES)
        return frozenset(str(item) for item in entry) & frozenset(TOOL_CAPABILITIES)
    return frozenset()


def referenced_capabilities(source: str) -> frozenset[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset(TOOL_CAPABILITIES)
    config_nodes = _config_literal_nodes(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", None)
            if name in _UNKNOWN_CALLS:
                return frozenset(TOOL_CAPABILITIES)
        if isinstance(node, ast.Attribute):
            capability = _METHOD_CAPABILITY.get(node.attr)
            if capability is not None:
                found.add(capability)
            capability = _TYPE_CAPABILITY.get(node.attr)
            if capability is not None:
                found.add(capability)
        if isinstance(node, ast.Name):
            capability = _TYPE_CAPABILITY.get(node.id)
            if capability is not None:
                found.add(capability)
        if isinstance(node, ast.Constant) and type(node.value) is str and id(node) not in config_nodes:
            if node.value in _METHOD_CAPABILITY.values() or node.value in _METHOD_CAPABILITY:
                found.add(_METHOD_CAPABILITY.get(node.value, node.value))
    return frozenset(found) & frozenset(TOOL_CAPABILITIES)


def capabilities_for_source(source: str) -> frozenset[str]:
    declared = declared_capabilities(source)
    wanted = referenced_capabilities(source) | required_capabilities(source)
    if declared is None:
        return frozenset(TOOL_CAPABILITIES)
    return frozenset(wanted & declared & frozenset(TOOL_CAPABILITIES))


def capabilities_for_scaffold(scaffold_dir: str | Path) -> frozenset[str]:
    path = Path(scaffold_dir)
    if path.is_dir():
        path = path / "scaffold.py"
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return frozenset(TOOL_CAPABILITIES)
    return capabilities_for_source(source)


__all__ = [
    "HEAVY_CAPABILITY",
    "TOOL_CAPABILITIES",
    "capabilities_for_scaffold",
    "capabilities_for_source",
    "declared_capabilities",
    "referenced_capabilities",
    "required_capabilities",
]
