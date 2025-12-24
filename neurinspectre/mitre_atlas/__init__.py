"""MITRE ATLAS (official) registry helpers.

This package vendors the official MITRE ATLAS STIX bundle for deterministic,
offline-safe taxonomy lookups.

Source bundle:
- https://github.com/mitre-atlas/atlas-navigator-data (dist/stix-atlas.json)
"""

from .registry import (
    AtlasTactic,
    AtlasTechnique,
    load_stix_atlas_bundle,
    list_atlas_tactics,
    list_atlas_techniques,
    techniques_by_tactic,
    tactic_by_phase_name,
    technique_index,
)

from .coverage import (
    AtlasValidationResult,
    ModuleAtlasCoverage,
    module_coverage,
    validate_atlas_ids,
    format_module_coverage_markdown,
    format_validation_json,
)
