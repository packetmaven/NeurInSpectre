"""MITRE ATLAS coverage + validation helpers.

This module is intentionally *offline-first*:
- It loads the official MITRE ATLAS STIX bundle vendored in-repo.
- It can validate that every AML.T*/AML.TA* ID referenced in NeurInSpectre code/docs
  exists in the official catalog.
- It can generate a per-module coverage report.

Notes on terminology:
- "Catalog support" means NeurInSpectre can list/show the technique/tactic from STIX.
- "Instrumentation" (detections/visualizations) varies by module; this report focuses
  on *references and mappings* present in the repository.

Authoritative sources:
- ATLAS taxonomy (release version): https://github.com/mitre-atlas/atlas-data (v5.1.1)
- STIX bundle used for offline lookups:
  https://github.com/mitre-atlas/atlas-navigator-data (dist/stix-atlas.json)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

import json
import re

from .registry import (
    list_atlas_tactics,
    load_stix_atlas_bundle,
    tactic_by_phase_name,
    technique_index,
)


_ATLAS_ID_RE = re.compile(r"\bAML\.(?:TA\d{4}|T\d{4})(?:\.\d{3})?\b")


@dataclass(frozen=True)
class AtlasValidationResult:
    tactic_count: int
    technique_count: int
    referenced_ids_total: int
    referenced_ids_unique: int
    unknown_ids: List[str]


@dataclass(frozen=True)
class ModuleAtlasCoverage:
    module_path: str
    # All AML.* IDs referenced in this module (tactics + techniques)
    atlas_ids: List[str]
    # Subsets (derived)
    tactic_ids: List[str]
    technique_ids: List[str]
    # Friendly names
    tactics: List[str]
    techniques: List[str]
    # Whether the module imports/loads the STIX registry (catalog-level support)
    catalog_support: bool


def neurinspectre_package_root() -> Path:
    """Return the local `neurinspectre/` package root as a Path."""
    return Path(__file__).resolve().parents[1]


def _iter_text_files(root: Path, *, exts: Sequence[str]) -> Iterable[Path]:
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        # Skip venvs/caches if present inside root
        parts = set(p.parts)
        if {'__pycache__', '.git', 'node_modules'} & parts:
            continue
        if any(part.startswith('venv') for part in p.parts):
            continue
        yield p


def extract_atlas_ids_from_text(text: str) -> List[str]:
    return _ATLAS_ID_RE.findall(text or '')


def scan_tree_for_atlas_ids(root: Path, *, exts: Sequence[str]) -> Dict[str, Set[str]]:
    """Scan files under root for AML.* IDs.

    Returns mapping: relative_path -> {ids...}
    """
    out: Dict[str, Set[str]] = {}
    for p in _iter_text_files(root, exts=exts):
        try:
            data = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        ids = set(extract_atlas_ids_from_text(data))
        if not ids:
            continue
        rel = str(p.relative_to(root))
        out[rel] = ids
    return out


def validate_atlas_ids(
    referenced_ids: Iterable[str],
    *,
    stix_bundle: Optional[dict] = None,
) -> AtlasValidationResult:
    """Validate a set of referenced AML.* IDs against the official STIX bundle."""
    bundle = stix_bundle or load_stix_atlas_bundle()

    # Build official ID set
    tactics = list_atlas_tactics(bundle)
    tech_idx = technique_index(bundle)
    official: Set[str] = {t.tactic_id for t in tactics} | set(tech_idx.keys())

    refs = list(referenced_ids)
    uniq = sorted(set(refs))
    unknown = sorted([i for i in uniq if i not in official])

    return AtlasValidationResult(
        tactic_count=len(tactics),
        technique_count=len(tech_idx),
        referenced_ids_total=len(refs),
        referenced_ids_unique=len(uniq),
        unknown_ids=unknown,
    )


def module_coverage(
    *,
    scope: Literal['code', 'docs', 'all'] = 'code',
    package_root: Optional[Path] = None,
) -> List[ModuleAtlasCoverage]:
    """Generate per-file/module ATLAS coverage.

    scope:
    - code: scan *.py under neurinspectre/
    - docs: scan *.md under repo root (best-effort)
    - all: both
    """
    pkg = package_root or neurinspectre_package_root()

    # Determine scan roots
    scan_items: List[Tuple[Path, Sequence[str]]] = []
    if scope in ('code', 'all'):
        scan_items.append((pkg, ('.py',)))
    if scope in ('docs', 'all'):
        # repo root is parent of neurinspectre/
        repo_root = pkg.parent
        scan_items.append((repo_root, ('.md',)))

    # Load STIX indexes once
    bundle = load_stix_atlas_bundle()
    tech_idx = technique_index(bundle)
    phase_to_tactic = tactic_by_phase_name(bundle)
    tactics = {t.tactic_id: t for t in list_atlas_tactics(bundle)}

    # Scan
    found: Dict[str, Set[str]] = {}
    for root, exts in scan_items:
        m = scan_tree_for_atlas_ids(root, exts=exts)
        # merge
        for rel, ids in m.items():
            key = str(Path(root.name) / rel) if root != pkg else str(Path('neurinspectre') / rel)
            found.setdefault(key, set()).update(ids)

    rows: List[ModuleAtlasCoverage] = []
    for rel, ids in sorted(found.items(), key=lambda x: x[0]):
        # Determine catalog_support (simple heuristic): module imports registry
        catalog_support = False
        abs_path = pkg.parent / rel
        try:
            if abs_path.is_file() and abs_path.suffix == '.py':
                txt = abs_path.read_text(encoding='utf-8', errors='ignore')
                catalog_support = 'mitre_atlas.registry' in txt
        except Exception:
            catalog_support = False

        tactic_ids = sorted([i for i in ids if i.startswith('AML.TA')])
        technique_ids = sorted([i for i in ids if i.startswith('AML.T') and not i.startswith('AML.TA')])

        # Expand technique -> tactic names
        tactic_names: Set[str] = set()
        technique_labels: List[str] = []

        for tid in tactic_ids:
            t = tactics.get(tid)
            if t is not None:
                tactic_names.add(t.name)

        for tech_id in technique_ids:
            tech = tech_idx.get(tech_id)
            if tech is None:
                continue
            tech_tactics = []
            for ph in tech.tactic_phase_names:
                t = phase_to_tactic.get(ph)
                if t is not None and t.name not in tech_tactics:
                    tech_tactics.append(t.name)
                    tactic_names.add(t.name)
            tactic_part = f" ({', '.join(tech_tactics)})" if tech_tactics else ''
            technique_labels.append(f"{tech.technique_id} {tech.name}{tactic_part}")

        rows.append(
            ModuleAtlasCoverage(
                module_path=rel,
                atlas_ids=sorted(ids),
                tactic_ids=tactic_ids,
                technique_ids=technique_ids,
                tactics=sorted(tactic_names),
                techniques=technique_labels,
                catalog_support=catalog_support,
            )
        )

    return rows


def format_module_coverage_markdown(rows: Sequence[ModuleAtlasCoverage]) -> str:
    """Render a GitHub-friendly nested <details> coverage block."""
    lines: List[str] = []
    lines.append('<details>')
    lines.append('<summary><b>MITRE ATLAS: per-module coverage (auto-generated)</b></summary>')
    lines.append('')
    lines.append('- **What this is**: a deterministic scan of NeurInSpectre source for `AML.T*` / `AML.TA*` references, normalized to the vendored STIX catalog.')
    lines.append('- **What it is not**: proof that every technique has a unique detector; many techniques are operational/behavioral and are supported via planning, mapping, and evaluation workflows.')
    lines.append('')

    for r in rows:
        tech_n = len(r.technique_ids)
        tac_n = len(r.tactic_ids)
        flags = []
        if r.catalog_support:
            flags.append('catalog')
        flag_txt = f" | {', '.join(flags)}" if flags else ''
        lines.append('<details>')
        lines.append(f"<summary><b>{r.module_path}</b> — {tech_n} techniques, {tac_n} tactics{flag_txt}</summary>")
        lines.append('')

        if r.tactics:
            lines.append('**Tactics (derived)**: ' + ', '.join(r.tactics))
            lines.append('')

        if r.tactic_ids:
            lines.append('**Tactic IDs referenced**: ' + ', '.join(r.tactic_ids))
            lines.append('')

        if r.techniques:
            lines.append('**Techniques referenced (STIX-normalized)**:')
            for t in r.techniques:
                lines.append(f'- {t}')
            lines.append('')

        lines.append('</details>')
        lines.append('')

    lines.append('</details>')
    lines.append('')
    return '\n'.join(lines)


def format_validation_json(res: AtlasValidationResult) -> str:
    return json.dumps(asdict(res), indent=2)

# ------------------------------
# Deeper auditing (labels + placeholders + upstream parity)
# ------------------------------


@dataclass(frozen=True)
class AtlasLabelMismatch:
    file_path: str
    line: int
    atlas_id: str
    found_label: str
    official_name: str
    kind: str  # e.g., 'paren' or 'bullet'


@dataclass(frozen=True)
class AtlasPlaceholderHit:
    file_path: str
    line: int
    pattern: str
    context: str


@dataclass(frozen=True)
class AtlasUpstreamParity:
    atlas_data_tag: str
    match: bool
    stix_ids: int
    upstream_ids: int
    only_in_stix: List[str]
    only_in_upstream: List[str]


def _official_id_to_name(*, stix_bundle: Optional[dict] = None) -> Dict[str, str]:
    b = stix_bundle or load_stix_atlas_bundle()
    out: Dict[str, str] = {}

    for t in list_atlas_tactics(b):
        out[t.tactic_id] = t.name

    tidx = technique_index(b)
    for tech_id, tech in tidx.items():
        out[tech_id] = tech.name

    return out


def scan_markdown_label_mismatches(
    repo_root: Path,
    *,
    stix_bundle: Optional[dict] = None,
) -> List[AtlasLabelMismatch]:
    """Find mismatches like `AML.Txxxx (Some label)` where label doesn't match STIX.

    This is intentionally strict: if a line uses an ATLAS ID as a label, it should include
    the official technique/tactic name to avoid drift.
    """
    b = stix_bundle or load_stix_atlas_bundle()
    id_to_name = _official_id_to_name(stix_bundle=b)

    paren_pat = re.compile(r"\b(AML\.(?:TA\d{4}|T\d{4})(?:\.\d{3})?)\s*\(([^\)]{3,140})\)")
    bullet_plain = re.compile(r"^\s*[-*]\s+(AML\.(?:TA\d{4}|T\d{4})(?:\.\d{3})?)\s+(.+?)\s*$")
    bullet_bold = re.compile(r"^\s*[-*]\s+\*\*(AML\.(?:TA\d{4}|T\d{4})(?:\.\d{3})?)\*\*\s*:\s*(.+?)\s*$")

    mismatches: List[AtlasLabelMismatch] = []

    for p in _iter_text_files(repo_root, exts=('.md',)):
        try:
            lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            continue

        for ln_no, line in enumerate(lines, start=1):
            # Parentheses patterns anywhere on the line
            for mid, label in paren_pat.findall(line):
                off = id_to_name.get(mid)
                if not off:
                    continue
                lab = re.sub(r"\s+", " ", label).strip()
                lab_l = lab.lower()
                off_l = str(off).strip().lower()
                if lab_l == off_l or off_l in lab_l or lab_l in off_l:
                    continue
                mismatches.append(
                    AtlasLabelMismatch(
                        file_path=str(p.relative_to(repo_root)),
                        line=int(ln_no),
                        atlas_id=str(mid),
                        found_label=lab,
                        official_name=str(off),
                        kind='paren',
                    )
                )

            # Bullet-labeled lines (strong form)
            m = bullet_plain.match(line)
            if m:
                mid, label = m.group(1), m.group(2)
                off = id_to_name.get(mid)
                if off:
                    lab = re.sub(r"\s+", " ", label).strip()
                    lab_l = lab.lower()
                    off_l = str(off).strip().lower()
                    if not (lab_l == off_l or off_l in lab_l or lab_l in off_l):
                        mismatches.append(
                            AtlasLabelMismatch(
                                file_path=str(p.relative_to(repo_root)),
                                line=int(ln_no),
                                atlas_id=str(mid),
                                found_label=lab,
                                official_name=str(off),
                                kind='bullet',
                            )
                        )
                continue

            m = bullet_bold.match(line)
            if m:
                mid, label = m.group(1), m.group(2)
                off = id_to_name.get(mid)
                if off:
                    lab = re.sub(r"\s+", " ", label).strip()
                    lab_l = lab.lower()
                    off_l = str(off).strip().lower()
                    if not (lab_l == off_l or off_l in lab_l or lab_l in off_l):
                        mismatches.append(
                            AtlasLabelMismatch(
                                file_path=str(p.relative_to(repo_root)),
                                line=int(ln_no),
                                atlas_id=str(mid),
                                found_label=lab,
                                official_name=str(off),
                                kind='bullet',
                            )
                        )

    return mismatches


def scan_tree_for_atlas_placeholders(
    root: Path,
    *,
    exts: Sequence[str] = ('.py', '.md', '.txt', '.json', '.sh'),
    patterns: Sequence[str] = (r"\bATLAS-(?:T\d{4}|TA\d{4}|[A-Z]{1,3}|T-[A-Z0-9]+)\b",),
) -> List[AtlasPlaceholderHit]:
    """Detect placeholder ATLAS IDs (e.g., 'ATLAS-...')."""
    hits: List[AtlasPlaceholderHit] = []

    for p in _iter_text_files(root, exts=exts):
        try:
            lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            continue

        for ln_no, line in enumerate(lines, start=1):
            for pat in patterns:
                try:
                    if re.search(pat, line):
                        ctx = line.strip()
                    else:
                        continue
                except re.error:
                    if pat not in line:
                        continue
                    ctx = line.strip()
                if len(ctx) > 180:
                    ctx = ctx[:177] + '...'
                hits.append(
                    AtlasPlaceholderHit(
                        file_path=str(p.relative_to(root)),
                        line=int(ln_no),
                        pattern=str(pat),
                        context=ctx,
                    )
                )

    return hits


def compare_catalog_to_atlas_data_yaml(
    *,
    atlas_data_tag: str = 'v5.1.1',
    stix_bundle: Optional[dict] = None,
    timeout_s: int = 30,
) -> AtlasUpstreamParity:
    """Optional online check: ensure our vendored STIX IDs match atlas-data dist/ATLAS.yaml."""
    b = stix_bundle or load_stix_atlas_bundle()
    id_to_name = _official_id_to_name(stix_bundle=b)
    stix_ids = set(id_to_name.keys())

    # Fetch upstream YAML
    import urllib.request

    url = f'https://raw.githubusercontent.com/mitre-atlas/atlas-data/{atlas_data_tag}/dist/ATLAS.yaml'
    req = urllib.request.Request(url, headers={'User-Agent': 'NeurInSpectre-atlas-verify'})
    with urllib.request.urlopen(req, timeout=int(timeout_s)) as r:
        yaml_text = r.read().decode('utf-8', errors='ignore')

    upstream_ids = set(_ATLAS_ID_RE.findall(yaml_text))

    only_in_stix = sorted(stix_ids - upstream_ids)
    only_in_upstream = sorted(upstream_ids - stix_ids)

    return AtlasUpstreamParity(
        atlas_data_tag=str(atlas_data_tag),
        match=(len(only_in_stix) == 0 and len(only_in_upstream) == 0),
        stix_ids=int(len(stix_ids)),
        upstream_ids=int(len(upstream_ids)),
        only_in_stix=only_in_stix,
        only_in_upstream=only_in_upstream,
    )

