"""MITRE ATLAS STIX registry.

We vendor the official `stix-atlas.json` bundle in-repo so NeurInSpectre can:
- List ALL 16 tactics and ALL 140 techniques (including sub-techniques)
- Map technique IDs ↔ names ↔ tactic phase(s)
- Keep README/docs synchronized with the authoritative framework

Data source: `mitre-atlas/atlas-navigator-data` (STIX 2.1 bundle)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json


STIX_ATLAS_BUNDLE_PATH = Path(__file__).with_name('stix-atlas.json')


@dataclass(frozen=True)
class AtlasTactic:
    tactic_id: str
    name: str
    phase_name: str
    description: str


@dataclass(frozen=True)
class AtlasTechnique:
    technique_id: str
    name: str
    description: str
    tactic_phase_names: List[str]
    url: Optional[str]
    is_subtechnique: bool


def _external_id(obj: dict) -> Optional[str]:
    for r in (obj.get('external_references') or []):
        if r.get('source_name') in ('mitre-atlas', 'mitre-atlas-techniques') and r.get('external_id'):
            return str(r.get('external_id'))
    return None


def _external_url(obj: dict) -> Optional[str]:
    for r in (obj.get('external_references') or []):
        if r.get('source_name') in ('mitre-atlas', 'mitre-atlas-techniques') and r.get('url'):
            return str(r.get('url'))
    return None


def load_stix_atlas_bundle(path: Optional[str | Path] = None) -> dict:
    p = Path(path) if path is not None else STIX_ATLAS_BUNDLE_PATH
    data = json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or 'objects' not in data:
        raise ValueError('Invalid STIX bundle: missing objects')
    return data


def list_atlas_tactics(bundle: Optional[dict] = None) -> List[AtlasTactic]:
    b = bundle or load_stix_atlas_bundle()
    tactics: List[AtlasTactic] = []
    for o in b.get('objects', []):
        if o.get('type') != 'x-mitre-tactic':
            continue
        tid = _external_id(o)
        if not tid:
            continue
        phase = str(o.get('x_mitre_shortname') or '').strip()
        desc = str(o.get('description') or '').strip()
        tactics.append(AtlasTactic(tactic_id=tid, name=str(o.get('name') or ''), phase_name=phase, description=desc))

    # stable sort by numeric tactic id
    def _k(t: AtlasTactic) -> Tuple[int, str]:
        try:
            n = int(str(t.tactic_id).split('AML.TA')[-1])
        except Exception:
            n = 10**9
        return (n, t.tactic_id)

    tactics.sort(key=_k)
    return tactics


def list_atlas_techniques(bundle: Optional[dict] = None) -> List[AtlasTechnique]:
    b = bundle or load_stix_atlas_bundle()
    techs: List[AtlasTechnique] = []
    for o in b.get('objects', []):
        if o.get('type') != 'attack-pattern':
            continue
        tid = _external_id(o)
        if not tid:
            continue

        phases: List[str] = []
        for p in (o.get('kill_chain_phases') or []):
            if p.get('kill_chain_name') == 'mitre-atlas' and p.get('phase_name'):
                phases.append(str(p.get('phase_name')))

        # MITRE ATLAS marks sub-techniques via x_mitre_is_subtechnique
        is_sub = bool(o.get('x_mitre_is_subtechnique', False))

        techs.append(
            AtlasTechnique(
                technique_id=tid,
                name=str(o.get('name') or ''),
                description=str(o.get('description') or '').strip(),
                tactic_phase_names=sorted(set(phases)),
                url=_external_url(o),
                is_subtechnique=is_sub,
            )
        )

    # sort: AML.T0000 < AML.T0000.000 < ... < AML.T0095
    def _parse_parts(s: str) -> Tuple[int, int]:
        try:
            body = s.split('AML.T', 1)[1]
            if '.' in body:
                a, b = body.split('.', 1)
                return (int(a), int(b))
            return (int(body), -1)
        except Exception:
            return (10**9, 10**9)

    techs.sort(key=lambda t: (_parse_parts(t.technique_id)[0], _parse_parts(t.technique_id)[1], t.technique_id))
    return techs


def tactic_by_phase_name(bundle: Optional[dict] = None) -> Dict[str, AtlasTactic]:
    # phase_name is the shortname used in kill_chain_phases (e.g. 'reconnaissance')
    b = bundle or load_stix_atlas_bundle()
    out: Dict[str, AtlasTactic] = {}
    for t in list_atlas_tactics(b):
        if t.phase_name:
            out[t.phase_name] = t
    return out


def techniques_by_tactic(bundle: Optional[dict] = None) -> Dict[str, List[AtlasTechnique]]:
    b = bundle or load_stix_atlas_bundle()
    phase_to_tactic = tactic_by_phase_name(b)

    by: Dict[str, List[AtlasTechnique]] = {t.tactic_id: [] for t in list_atlas_tactics(b)}
    for tech in list_atlas_techniques(b):
        for ph in tech.tactic_phase_names:
            tactic = phase_to_tactic.get(ph)
            if tactic is None:
                continue
            by.setdefault(tactic.tactic_id, []).append(tech)

    for k in list(by.keys()):
        by[k].sort(key=lambda t: t.technique_id)
    return by


def technique_index(bundle: Optional[dict] = None) -> Dict[str, AtlasTechnique]:
    b = bundle or load_stix_atlas_bundle()
    return {t.technique_id: t for t in list_atlas_techniques(b)}
