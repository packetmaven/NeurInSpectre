"""NeurInSpectre MITRE ATLAS CLI (offline STIX).

This exposes:
- listing tactics/techniques
- showing technique/tactic details
- per-module coverage reports (code/doc scanning)
- validation (unknown IDs, catalog size)

Catalog source:
- ATLAS taxonomy: https://github.com/mitre-atlas/atlas-data (v5.1.1)
- Offline STIX bundle: https://github.com/mitre-atlas/atlas-navigator-data (dist/stix-atlas.json)
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..mitre_atlas.registry import (
    list_atlas_tactics,
    list_atlas_techniques,
    load_stix_atlas_bundle,
    tactic_by_phase_name,
    technique_index,
)

from ..mitre_atlas.coverage import (
    module_coverage,
    validate_atlas_ids,
    scan_markdown_label_mismatches,
    scan_tree_for_atlas_placeholders,
    compare_catalog_to_atlas_data_yaml,
    neurinspectre_package_root,
)


def register_mitre_atlas(subparsers) -> None:
    atlas = subparsers.add_parser(
        'mitre-atlas',
        help='MITRE ATLAS catalog + coverage tools (offline STIX)',
    )
    atlas_sub = atlas.add_subparsers(dest='atlas_action', required=True)

    # list
    lp = atlas_sub.add_parser('list', help='List ATLAS tactics/techniques')
    lp_sub = lp.add_subparsers(dest='list_kind', required=True)

    lpt = lp_sub.add_parser('tactics', help='List all tactics (16)')
    lpt.add_argument('--format', choices=['table', 'json', 'markdown'], default='table')

    lpx = lp_sub.add_parser('techniques', help='List all techniques (140; includes sub-techniques)')
    lpx.add_argument('--format', choices=['table', 'json', 'markdown'], default='table')
    lpx.add_argument(
        '--tactic',
        default=None,
        help='Filter by tactic id (AML.TA####) or tactic name (case-insensitive)',
    )
    lpx.add_argument(
        '--no-subtechniques',
        action='store_true',
        help='Exclude sub-techniques (AML.Txxxx.yyy)',
    )

    # show
    sp = atlas_sub.add_parser('show', help='Show details for a single AML.T*/AML.TA* id')
    sp.add_argument('atlas_id', help='ATLAS ID (e.g., AML.T0051 or AML.TA0005)')
    sp.add_argument('--format', choices=['text', 'json'], default='text')

    # coverage
    cp = atlas_sub.add_parser('coverage', help='Per-module ATLAS coverage (scan NeurInSpectre source)')
    cp.add_argument('--scope', choices=['code', 'docs', 'all'], default='code')
    cp.add_argument('--format', choices=['markdown', 'json'], default='markdown')
    cp.add_argument('--out', default=None, help='Write output to file instead of stdout')

    # validate
    vp = atlas_sub.add_parser('validate', help='Validate ATLAS catalog + references')
    vp.add_argument('--scope', choices=['code', 'docs', 'all'], default='all')
    vp.add_argument('--format', choices=['json', 'text'], default='json')
    vp.add_argument(
        '--strict',
        action='store_true',
        help='Exit non-zero if any unknown AML.* IDs are found',
    )

    vp.add_argument(
        '--verify-upstream',
        action='store_true',
        help='Online check: verify vendored STIX IDs match mitre-atlas/atlas-data dist/ATLAS.yaml',
    )
    vp.add_argument(
        '--atlas-data-tag',
        default='v5.1.1',
        help='atlas-data release tag for --verify-upstream (default: v5.1.1)',
    )


def _print_or_write(s: str, out_path: Optional[str]) -> None:
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s, encoding='utf-8')
        print(str(p))
    else:
        print(s)


def run_mitre_atlas(args) -> int:
    bundle = load_stix_atlas_bundle()

    if args.atlas_action == 'list':
        if args.list_kind == 'tactics':
            tactics = list_atlas_tactics(bundle)
            if args.format == 'json':
                print(json.dumps([asdict(t) for t in tactics], indent=2))
                return 0
            if args.format == 'markdown':
                for t in tactics:
                    print(f"- {t.tactic_id} {t.name}")
                return 0

            # table
            print('tactic_id	name	phase_name')
            for t in tactics:
                print(f"{t.tactic_id}	{t.name}	{t.phase_name}")
            return 0

        if args.list_kind == 'techniques':
            techs = list_atlas_techniques(bundle)

            # Filter by tactic
            if getattr(args, 'tactic', None):
                q = str(args.tactic).strip().lower()
                phase_to_tactic = tactic_by_phase_name(bundle)
                tactics = list_atlas_tactics(bundle)
                id_to_t = {t.tactic_id.lower(): t for t in tactics}

                want_tactic_id = None
                if q.startswith('aml.ta'):
                    want_tactic_id = q
                else:
                    # match by tactic name
                    for t in tactics:
                        if t.name.lower() == q:
                            want_tactic_id = t.tactic_id.lower()
                            break

                if want_tactic_id:
                    def _tech_has_tactic(tech) -> bool:
                        for ph in tech.tactic_phase_names:
                            tt = phase_to_tactic.get(ph)
                            if tt and tt.tactic_id.lower() == want_tactic_id:
                                return True
                        return False

                    techs = [t for t in techs if _tech_has_tactic(t)]

            if getattr(args, 'no_subtechniques', False):
                techs = [t for t in techs if ('.' not in t.technique_id and not t.is_subtechnique)]

            if args.format == 'json':
                print(json.dumps([asdict(t) for t in techs], indent=2))
                return 0

            if args.format == 'markdown':
                for t in techs:
                    print(f"- {t.technique_id} {t.name}")
                return 0

            # table
            phase_to_tactic = tactic_by_phase_name(bundle)
            print('technique_id	name	tactics')
            for t in techs:
                tnames = []
                for ph in t.tactic_phase_names:
                    tac = phase_to_tactic.get(ph)
                    if tac and tac.name not in tnames:
                        tnames.append(tac.name)
                print(f"{t.technique_id}	{t.name}	{', '.join(tnames)}")
            return 0

        return 1

    if args.atlas_action == 'show':
        atlas_id = str(args.atlas_id).strip()
        tactics = list_atlas_tactics(bundle)
        tidx = technique_index(bundle)
        phase_to_tactic = tactic_by_phase_name(bundle)

        obj: Dict[str, Any]
        if atlas_id.startswith('AML.TA'):
            t = next((x for x in tactics if x.tactic_id == atlas_id), None)
            if t is None:
                print(f"Unknown tactic id: {atlas_id}")
                return 1
            obj = {
                'id': t.tactic_id,
                'type': 'tactic',
                'name': t.name,
                'phase_name': t.phase_name,
                'description': t.description,
            }
        else:
            tech = tidx.get(atlas_id)
            if tech is None:
                print(f"Unknown technique id: {atlas_id}")
                return 1
            tnames = []
            tids = []
            for ph in tech.tactic_phase_names:
                tac = phase_to_tactic.get(ph)
                if tac and tac.tactic_id not in tids:
                    tids.append(tac.tactic_id)
                    tnames.append(tac.name)
            obj = {
                'id': tech.technique_id,
                'type': 'technique',
                'name': tech.name,
                'tactic_ids': tids,
                'tactics': tnames,
                'tactic_phase_names': tech.tactic_phase_names,
                'url': tech.url,
                'is_subtechnique': tech.is_subtechnique,
                'description': tech.description,
            }

        if args.format == 'json':
            print(json.dumps(obj, indent=2))
            return 0

        # text
        print(f"{obj['id']} ({obj['type']})")
        print(f"Name: {obj.get('name')}")
        if obj.get('type') == 'tactic':
            print(f"Phase: {obj.get('phase_name')}")
        else:
            print(f"Tactics: {', '.join(obj.get('tactics') or [])}")
            if obj.get('url'):
                print(f"URL: {obj.get('url')}")
        desc = (obj.get('description') or '').strip()
        if desc:
            print('')
            print(desc)
        return 0

    if args.atlas_action == 'coverage':
        rows = module_coverage(scope=str(getattr(args, 'scope', 'code')))

        if args.format == 'json':
            payload = [asdict(r) for r in rows]
            _print_or_write(json.dumps(payload, indent=2), getattr(args, 'out', None))
            return 0

        from ..mitre_atlas.coverage import format_module_coverage_markdown

        md = format_module_coverage_markdown(rows)
        _print_or_write(md, getattr(args, 'out', None))
        return 0

    if args.atlas_action == 'validate':
        scope = str(getattr(args, 'scope', 'all'))
        rows = module_coverage(scope=scope)
        all_ids: List[str] = []
        for r in rows:
            all_ids.extend(list(r.atlas_ids))

        res = validate_atlas_ids(all_ids, stix_bundle=bundle)

        # Deep checks
        pkg = neurinspectre_package_root()
        repo_root = pkg.parent

        label_mismatches = []
        if scope in ('docs', 'all'):
            try:
                label_mismatches = scan_markdown_label_mismatches(repo_root, stix_bundle=bundle)
            except Exception:
                label_mismatches = []

        placeholder_hits = []
        try:
            if scope == 'code':
                placeholder_hits = scan_tree_for_atlas_placeholders(pkg, exts=('.py',), patterns=('ATLAS-',))
            elif scope == 'docs':
                placeholder_hits = scan_tree_for_atlas_placeholders(repo_root, exts=('.md',), patterns=('ATLAS-',))
            else:
                placeholder_hits = scan_tree_for_atlas_placeholders(repo_root, exts=('.py', '.md'), patterns=('ATLAS-',))
        except Exception:
            placeholder_hits = []

        upstream = None
        if bool(getattr(args, 'verify_upstream', False)):
            try:
                upstream = compare_catalog_to_atlas_data_yaml(
                    atlas_data_tag=str(getattr(args, 'atlas_data_tag', 'v5.1.1')),
                    stix_bundle=bundle,
                )
            except Exception as e:
                upstream = {'error': str(e)}

        if args.format == 'text':
            print(f"tactics={res.tactic_count} techniques={res.technique_count}")
            print(f"referenced_unique={res.referenced_ids_unique} unknown={len(res.unknown_ids)}")
            print(f"label_mismatches={len(label_mismatches)} placeholders={len(placeholder_hits)}")
            if upstream is not None:
                try:
                    m = bool(getattr(upstream, 'match'))
                    print(f"upstream_match={m}")
                except Exception:
                    print("upstream_match=error")

            if res.unknown_ids:
                print('unknown_ids:')
                for x in res.unknown_ids:
                    print(' -', x)

            if label_mismatches:
                print('label_mismatches:')
                for mm in label_mismatches[:50]:
                    try:
                        print(f" - {mm.file_path}:{mm.line} {mm.atlas_id} label='{mm.found_label}' official='{mm.official_name}'")
                    except Exception:
                        pass

            if placeholder_hits:
                print('placeholders:')
                for hh in placeholder_hits[:50]:
                    try:
                        print(f" - {hh.file_path}:{hh.line} {hh.pattern} {hh.context}")
                    except Exception:
                        pass

        else:
            payload = asdict(res)
            payload['label_mismatches'] = [asdict(x) for x in label_mismatches]
            payload['placeholder_hits'] = [asdict(x) for x in placeholder_hits]
            if upstream is not None:
                try:
                    payload['upstream'] = asdict(upstream)  # dataclass
                except Exception:
                    payload['upstream'] = upstream
            print(json.dumps(payload, indent=2))

        strict = bool(getattr(args, 'strict', False))
        if strict:
            if res.unknown_ids:
                return 1
            if label_mismatches:
                return 1
            if placeholder_hits:
                return 1
            if upstream is not None:
                try:
                    if hasattr(upstream, 'match') and not bool(upstream.match):
                        return 1
                except Exception:
                    return 1

        return 0

    print('Unknown mitre-atlas action')
    return 1
