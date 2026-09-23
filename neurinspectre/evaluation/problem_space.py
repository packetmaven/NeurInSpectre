"""Problem-space validity helpers.

PE parse (pefile) and overlay-append live here. Full DOS / padding functionality
gates live in ``neurinspectre.malware.pe_transforms``. A success that fails the
gate must not be counted as a security finding. GAMMA *section* injection is
not implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union


def evaluate_pe_parse(
    source: Optional[Union[str, Path, bytes]] = None,
    *,
    enabled: bool = True,
) -> Dict[str, Any]:
    """Parse a PE with ``pefile`` if available.

    Returns a structured report. Missing ``pefile`` or a missing sample is
    recorded, not raised — audit should still complete.
    """
    if not enabled:
        return {"enabled": False, "kind": "pe_parse"}
    if source is None:
        return {
            "enabled": True,
            "kind": "pe_parse",
            "available": False,
            "passed": None,
            "reasons": ["no_pe_sample"],
            "hint": "Pass --pe-sample path/to/file.exe to run the Month 2 parse gate.",
        }

    try:
        import pefile  # type: ignore
    except ImportError:
        return {
            "enabled": True,
            "kind": "pe_parse",
            "available": False,
            "passed": None,
            "reasons": ["pefile_not_installed"],
            "hint": "pip install pefile  (Month 3 will also need secml-malware)",
        }

    data: Optional[bytes] = None
    path_str = None
    if isinstance(source, (str, Path)):
        path = Path(source)
        path_str = str(path)
        if not path.is_file():
            return {
                "enabled": True,
                "kind": "pe_parse",
                "available": True,
                "passed": False,
                "reasons": ["pe_sample_missing"],
                "path": path_str,
            }
        data = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        return {
            "enabled": True,
            "kind": "pe_parse",
            "available": True,
            "passed": False,
            "reasons": ["unsupported_pe_source"],
        }

    try:
        pe = pefile.PE(data=data)
        n_sections = len(getattr(pe, "sections", []) or [])
        entry = int(getattr(getattr(pe, "OPTIONAL_HEADER", None), "AddressOfEntryPoint", 0) or 0)
        machine = int(getattr(getattr(pe, "FILE_HEADER", None), "Machine", 0) or 0)
        passed = bool(data[:2] == b"MZ" and n_sections >= 1)
        reasons = [] if passed else ["pe_parse_incomplete"]
        return {
            "enabled": True,
            "kind": "pe_parse",
            "available": True,
            "passed": passed,
            "reasons": reasons,
            "path": path_str,
            "observed": {
                "mz": data[:2] == b"MZ",
                "machine": machine,
                "entry_point": entry,
                "n_sections": n_sections,
                "size_bytes": len(data),
            },
        }
    except Exception as exc:
        return {
            "enabled": True,
            "kind": "pe_parse",
            "available": True,
            "passed": False,
            "reasons": ["pe_parse_failed"],
            "path": path_str,
            "error": str(exc),
        }


def evaluate_pe_functionality(
    original: Optional[bytes],
    mutated: Optional[bytes],
    *,
    expect_overlay_only: bool = True,
) -> Dict[str, Any]:
    """Compare original vs mutated PE. Overlay-append must keep headers/EP/sections."""
    if original is None or mutated is None:
        return {
            "enabled": True,
            "kind": "pe_functionality",
            "passed": None,
            "reasons": ["missing_pe_bytes"],
        }
    before = evaluate_pe_parse(original)
    after = evaluate_pe_parse(mutated)
    reasons: list = []
    if not before.get("available"):
        return {
            "enabled": True,
            "kind": "pe_functionality",
            "passed": None,
            "reasons": list(before.get("reasons") or ["pe_parse_unavailable"]),
            "before": before,
            "after": after,
        }
    if before.get("passed") is not True:
        reasons.append("original_pe_invalid")
    if after.get("passed") is not True:
        reasons.append("mutated_pe_invalid")
    b_obs = dict(before.get("observed") or {})
    a_obs = dict(after.get("observed") or {})
    if b_obs.get("entry_point") != a_obs.get("entry_point"):
        reasons.append("entry_point_changed")
    if expect_overlay_only and b_obs.get("n_sections") != a_obs.get("n_sections"):
        reasons.append("section_count_changed")
    if expect_overlay_only and len(mutated) < len(original):
        reasons.append("file_shrunk")
    if expect_overlay_only and mutated[: len(original)] != original:
        reasons.append("original_bytes_rewritten")
    return {
        "enabled": True,
        "kind": "pe_functionality",
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "before": before,
        "after": after,
        "size_delta": int(len(mutated) - len(original)),
    }


def overlay_append(pe_bytes: bytes, payload: bytes) -> bytes:
    """Append bytes after the PE image. Headers, sections, and EP stay intact."""
    if not pe_bytes:
        raise ValueError("overlay_append requires PE bytes")
    return bytes(pe_bytes) + bytes(payload or b"")


def compute_asr_query_curve(
    queries: list,
    success: list,
    budgets: list,
) -> list:
    """ASR at each query budget from per-sample query counts.

    ``n`` is the number of clean-correct samples that were attacked.
    A sample counts as a success at budget Q if it was adversarial and
    ``queries_used <= Q``. The initial model evaluation is query 1.
    """
    if not queries or not success or len(queries) != len(success):
        return []
    n = len(success)
    curve = []
    for raw in budgets:
        try:
            q = int(raw)
        except (TypeError, ValueError):
            continue
        if q <= 0:
            continue
        hits = sum(1 for qi, s in zip(queries, success) if bool(s) and int(qi) <= q)
        curve.append(
            {
                "query_budget": q,
                "asr": float(hits / n) if n else 0.0,
                "successes": int(hits),
                "n": int(n),
            }
        )
    return curve
