"""Problem-space PE attacks for EMBER GBDT audits.

Implemented here (literature names, not a secml-malware wrap):
  - Full DOS (Demetrio et al.)
  - Padding / overlay (Kolosnjaji et al.)
  - GAMMA-padding when the payload is copied from benign PEs

GAMMA *section* injection is not implemented. If secml-malware is installed we
only record that fact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from ..evaluation.problem_space import evaluate_pe_parse, overlay_append
from ..malware.ember_extract import extract_ember_features
from ..malware.pe_transforms import (
    apply_combined_multi_region,
    apply_fulldos,
    apply_padding,
    apply_section_slack_pad,
    evaluate_transform_validity,
    fulldos_capacity,
    section_slack_capacity,
)


def try_import_secml_gamma() -> Optional[Any]:
    try:
        from secml_malware.attack.blackbox.c_gamma_sections_evasion import (  # type: ignore
            CGammaSectionsEvasionAttack,
        )

        return CGammaSectionsEvasionAttack
    except Exception:
        return None


def predict_malware(model, features) -> tuple:
    """Return (pred, p_malware) from 2-class logits or a binary score."""
    feats = torch.as_tensor(np.asarray(features, dtype=np.float32)).reshape(1, -1)
    with torch.no_grad():
        logits = model(feats)
    if logits.ndim == 1 or (logits.ndim == 2 and int(logits.size(1)) == 1):
        score = float(logits.view(-1)[0].item())
        pred = int(score > 0)
        p_mal = float(1.0 / (1.0 + np.exp(-score)))
        return pred, p_mal
    logits_np = logits.detach().cpu().numpy().reshape(-1)
    pred = int(np.argmax(logits_np))
    m = float(np.max(logits_np))
    exps = np.exp(logits_np - m)
    p_mal = float(exps[1] / exps.sum()) if exps.size > 1 else float(exps[0])
    return pred, p_mal


_PE_SUFFIXES = {".exe", ".dll", ".sys", ".ocx", ".scr", ".cpl", ".efi", ".acm", ".ax"}


def _iter_pe_paths(root: Path) -> List[Path]:
    """Non-hidden files; recurse for common PE suffixes so nested corpora work."""
    if root.is_file():
        return [root]
    found: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        suffix = path.suffix.lower()
        if suffix in _PE_SUFFIXES or path.parent == root:
            found.append(path)
    return sorted(found)


def load_pe_samples(source: Optional[Union[str, Path, bytes, Sequence[Union[str, Path, bytes]]]]) -> List[Dict[str, Any]]:
    if source is None:
        return []
    items: List[Union[str, Path, bytes]]
    if isinstance(source, (bytes, bytearray)):
        items = [bytes(source)]
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            items = _iter_pe_paths(path)
        else:
            items = [path]
    else:
        items = list(source)
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, (bytes, bytearray)):
            out.append({"path": None, "bytes": bytes(item)})
            continue
        path = Path(item)
        if not path.is_file():
            out.append({"path": str(path), "bytes": None, "error": "pe_sample_missing"})
            continue
        out.append(_read_pe_candidate(path))
    return out


def _read_pe_candidate(path: Path) -> Dict[str, Any]:
    """Load a file only if it starts with MZ. Mixed directories are not all PEs."""
    try:
        with path.open("rb") as handle:
            head = handle.read(64)
    except OSError as exc:
        return {"path": str(path), "bytes": None, "error": f"pe_unreadable:{exc}"}
    if len(head) < 2 or head[:2] != b"MZ":
        return {"path": str(path), "bytes": None, "error": "not_a_valid_pe"}
    return {"path": str(path), "bytes": path.read_bytes()}


def load_benign_payloads(source: Optional[Union[str, Path]]) -> List[bytes]:
    if source is None:
        return []
    samples = load_pe_samples(source)
    return [s["bytes"] for s in samples if s.get("bytes")]


class OverlayAppendAttack:
    """Single-shot overlay. Kept for Month 2 tests."""

    def __init__(
        self,
        model=None,
        *,
        payload: bytes = b"\x00" * 256,
        extractor: Optional[Callable[[bytes], Dict[str, Any]]] = None,
        n_queries: int = 1,
    ):
        self.model = model
        self.payload = bytes(payload)
        self.extractor = extractor
        self.n_queries = int(n_queries)
        self.chosen_attack = "overlay_append"
        self.selected_attack_impl = "OverlayAppendAttack"

    def run_bytes(self, pe_bytes: bytes, y: Optional[int] = None) -> Dict[str, Any]:
        parse_before = evaluate_pe_parse(pe_bytes)
        mutated = overlay_append(pe_bytes, self.payload)
        func = evaluate_transform_validity(pe_bytes, mutated, kind="overlay")
        extracted = extract_ember_features(mutated, extractor=self.extractor)
        scored = False
        pred = None
        success = None
        if extracted.get("features") is not None and self.model is not None:
            pred, _p = predict_malware(self.model, extracted["features"])
            scored = True
            if y is not None:
                success = bool(pred != int(y))
        return {
            "chosen_attack": self.chosen_attack,
            "selected_attack_impl": self.selected_attack_impl,
            "space": "problem",
            "realizable": bool(func.get("passed")),
            "parse_before": parse_before,
            "functionality": func,
            "extraction": {k: v for k, v in extracted.items() if k != "features"},
            "scored": scored,
            "prediction": pred,
            "success": success,
            "queries_used": 1,
            "payload_bytes": len(self.payload),
            "gamma": False,
            "note": "overlay_append is padding, not GAMMA section injection.",
        }


def evaluate_overlay_corpus(
    samples: Sequence[Dict[str, Any]],
    model=None,
    *,
    payload: bytes = b"\x00" * 256,
    labels: Optional[Sequence[int]] = None,
    extractor: Optional[Callable[[bytes], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    attack = OverlayAppendAttack(model, payload=payload, extractor=extractor)
    rows = []
    for i, sample in enumerate(samples):
        data = sample.get("bytes")
        y = None if labels is None or i >= len(labels) else int(labels[i])
        if data is None:
            rows.append(
                {
                    "path": sample.get("path"),
                    "success": None,
                    "realizable": False,
                    "scored": False,
                    "reasons": [sample.get("error") or "pe_sample_missing"],
                }
            )
            continue
        row = attack.run_bytes(data, y=y)
        row["path"] = sample.get("path")
        rows.append(row)
    scored = [r for r in rows if r.get("scored") and r.get("success") is not None]
    valid = [r for r in rows if r.get("realizable")]
    valid_success = [r for r in scored if r.get("success") and r.get("realizable")]
    n_scored = len(scored)
    return {
        "kind": "problem_space_overlay",
        "n_samples": len(rows),
        "n_scored": n_scored,
        "n_valid": len(valid),
        "attack_success_rate": (sum(1 for r in scored if r.get("success")) / n_scored) if n_scored else None,
        "valid_success_rate": (len(valid_success) / n_scored) if n_scored else None,
        "secml_gamma_available": try_import_secml_gamma() is not None,
        "samples": rows,
    }


class ProblemSpacePESearch:
    """Query-limited random search over Full DOS and padding."""

    def __init__(
        self,
        model,
        *,
        n_queries: int = 50,
        payload_size: int = 256,
        seed: int = 42,
        benign_payloads: Optional[Sequence[bytes]] = None,
        extractor: Optional[Callable[[bytes], Dict[str, Any]]] = None,
        capa_preserve: bool = False,
        capa_rules_dir: Optional[Path] = None,
        capa_preserve_mode: str = "all",
        enable_section_slack: bool = False,
        transform_set: str = "default",
        supplement_payloads: Optional[Sequence[bytes]] = None,
        fulldos_quiet_only: bool = False,
    ):
        self.model = model
        self.n_queries = int(n_queries)
        self.payload_size = int(payload_size)
        self.seed = int(seed)
        self.benign_payloads = [bytes(p) for p in (benign_payloads or []) if p]
        self.extractor = extractor
        self.chosen_attack = "problem_space"
        self.selected_attack_impl = "ProblemSpacePESearch"
        # C7 — section-slack transform + supplement-shaped payloads
        self.enable_section_slack = bool(enable_section_slack)
        self.supplement_payloads = [bytes(p) for p in (supplement_payloads or []) if p]
        self.n_section_slack_attempts: int = 0
        self.n_section_slack_no_capacity: int = 0
        # D10 — transform_set switches the mutation mixture.
        # "default":  the current mix (Full DOS + padding, plus section_slack
        #             when enable_section_slack=True).
        # "combined": AdvMal-TF / PhantomCall multi-region envelope — every
        #             candidate applies Full DOS + section slack + overlay
        #             in one shot. Also implies enable_section_slack.
        self.transform_set = str(transform_set or "default").lower()
        if self.transform_set not in {"default", "combined"}:
            raise ValueError(
                f"transform_set must be 'default' or 'combined', got {transform_set!r}"
            )
        if self.transform_set == "combined":
            self.enable_section_slack = True
        self.n_combined_attempts: int = 0
        # E12 — reject candidates whose pefilewarnings feature band differs
        # from the baseline. Empirically this is very rare on our corpus,
        # but the gate is here to make the "quiet" property a hard invariant
        # rather than an accidental one.
        self.fulldos_quiet_only = bool(fulldos_quiet_only)
        self._pefw_baseline: Optional[np.ndarray] = None
        self.n_pefw_rejects: int = 0
        # C6 + C8 — capability preservation gate (file-level Capa).
        # C8 mode selects the strictness: all/ttps/mbc.
        self.capa_preserve = bool(capa_preserve)
        self.capa_rules_dir = capa_rules_dir
        self.capa_preserve_mode = str(capa_preserve_mode or "all").lower()
        if self.capa_preserve_mode not in {"all", "ttps", "mbc"}:
            raise ValueError(
                f"capa_preserve_mode must be one of all/ttps/mbc, got {capa_preserve_mode!r}"
            )
        self._capa_ok: Optional[bool] = None   # cached availability
        self._capa_error: Optional[str] = None
        self.n_capa_rejects: int = 0
        self.n_capa_calls: int = 0

    def _capa_baseline(self, pe_bytes: bytes):
        """Compute the capability set (or restricted subset per ``capa_preserve_mode``)
        of the original bytes. None if disabled/unavailable."""
        if not self.capa_preserve:
            return None
        try:
            from ..malware.capa_scan import (
                capabilities_file_level, CapaUnavailable,
                restrict_to_ttps, restrict_to_mbc,
            )
        except ImportError as exc:
            self._capa_ok = False
            self._capa_error = f"capa_scan_import_failed: {exc}"
            return None
        try:
            caps = capabilities_file_level(pe_bytes, rules_dir=self.capa_rules_dir)
            if self.capa_preserve_mode == "ttps":
                caps = restrict_to_ttps(caps, rules_dir=self.capa_rules_dir)
            elif self.capa_preserve_mode == "mbc":
                caps = restrict_to_mbc(caps, rules_dir=self.capa_rules_dir)
            self._capa_ok = True
            return caps
        except Exception as exc:
            self._capa_ok = False
            self._capa_error = f"{type(exc).__name__}: {exc}"
            return None

    def _capa_preserved(self, baseline, mutated_bytes: bytes) -> bool:
        """True iff the mutation keeps every capability the baseline exhibited
        (or the ``capa_preserve_mode`` subset)."""
        if baseline is None:
            return True
        try:
            from ..malware.capa_scan import (
                capabilities_file_level, restrict_to_ttps, restrict_to_mbc,
            )
        except ImportError:
            return True
        try:
            self.n_capa_calls += 1
            mutated_caps = capabilities_file_level(mutated_bytes, rules_dir=self.capa_rules_dir)
            if self.capa_preserve_mode == "ttps":
                mutated_caps = restrict_to_ttps(mutated_caps, rules_dir=self.capa_rules_dir)
            elif self.capa_preserve_mode == "mbc":
                mutated_caps = restrict_to_mbc(mutated_caps, rules_dir=self.capa_rules_dir)
        except Exception:
            # If Capa fails on a mutated file (corruption etc.), treat as capability loss.
            self.n_capa_rejects += 1
            return False
        if baseline - mutated_caps:
            self.n_capa_rejects += 1
            return False
        return True

    def _payload(self, rng: np.random.Generator, n: int) -> bytes:
        if self.benign_payloads:
            src = self.benign_payloads[int(rng.integers(0, len(self.benign_payloads)))]
            if len(src) >= n:
                start = int(rng.integers(0, max(1, len(src) - n + 1)))
                return src[start : start + n]
            return src + bytes(n - len(src))
        return rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()

    def _supplement_or_random_payload(self, rng: np.random.Generator, n: int) -> bytes:
        """Prefer bytes carved from Capa supplement functions when available."""
        if self.supplement_payloads:
            src = self.supplement_payloads[int(rng.integers(0, len(self.supplement_payloads)))]
            if len(src) >= n:
                start = int(rng.integers(0, max(1, len(src) - n + 1)))
                return src[start : start + n]
            return src + bytes(n - len(src))
        return self._payload(rng, n)

    def run_bytes(
        self,
        pe_bytes: bytes,
        y: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        # Counters are per file. A reused search object must not report a
        # running total from the previous PE.
        self.n_capa_calls = 0
        self.n_capa_rejects = 0
        self.n_pefw_rejects = 0
        self.n_section_slack_attempts = 0
        self.n_section_slack_no_capacity = 0
        self.n_combined_attempts = 0
        rng = rng or np.random.default_rng(self.seed)
        extracted0 = extract_ember_features(pe_bytes, extractor=self.extractor)
        if extracted0.get("features") is None:
            return {
                "chosen_attack": self.chosen_attack,
                "success": False,
                "realizable": False,
                "scored": False,
                "queries_used": 0,
                "reasons": list(extracted0.get("reasons") or ["extract_failed"]),
            }
        pred0, p0 = predict_malware(self.model, extracted0["features"])
        queries = 1
        success = bool(pred0 != int(y))
        best_p = p0
        best_kind = "clean"
        best_valid = True
        if success:
            return {
                "chosen_attack": "clean_already_evading",
                "selected_attack_impl": self.selected_attack_impl,
                "space": "problem",
                "success": True,
                "realizable": True,
                "scored": True,
                "queries_used": 1,
                "clean_p_malware": p0,
                "best_p_malware": p0,
                "gamma": False,
            }

        try:
            capacity = fulldos_capacity(pe_bytes)
        except ValueError:
            capacity = 0
        slack_cap = section_slack_capacity(pe_bytes, -1) if self.enable_section_slack else 0
        # C6 — compute Capa baseline once so per-mutation gating is a set diff.
        capa_baseline = self._capa_baseline(pe_bytes)
        # E12 — pefilewarnings baseline (slice out of the clean vector).
        if self.fulldos_quiet_only and extracted0.get("features") is not None:
            try:
                from ..malware.ember2024_extract import pefilewarnings_offset_dim
                off, dim = pefilewarnings_offset_dim()
                self._pefw_baseline = np.asarray(
                    extracted0["features"], dtype=np.float32
                )[off : off + dim].copy()
            except Exception:
                self._pefw_baseline = None
        # D9 — remember the best-of-search candidate bytes so downstream
        # tooling can re-score them against other detectors.
        best_bytes: bytes = pe_bytes
        history = []
        for _step in range(max(self.n_queries - 1, 0)):
            # D10 — combined transform set: every candidate is a
            # multi-region mutation (Full DOS + section slack + overlay).
            if self.transform_set == "combined":
                mode = "combined_multi_region"
            # C7 — three-way transform choice when section slack is enabled;
            # otherwise fall back to the original 50/50 Full-DOS/padding coin.
            elif self.enable_section_slack and slack_cap > 0:
                r = float(rng.random())
                if capacity > 0 and r < 1 / 3:
                    mode = "fulldos"
                elif r < 2 / 3:
                    mode = "section_slack"
                else:
                    mode = "padding"
            else:
                mode = "fulldos" if (capacity > 0 and rng.random() < 0.5) else "padding"

            if mode == "fulldos":
                kind = "fulldos"
                try:
                    cand = apply_fulldos(pe_bytes, self._supplement_or_random_payload(rng, capacity))
                except ValueError:
                    history.append({"kind": kind, "valid": False, "scored": False})
                    queries += 1
                    continue
            elif mode == "section_slack":
                kind = "section_slack"
                self.n_section_slack_attempts += 1
                if slack_cap <= 0:
                    self.n_section_slack_no_capacity += 1
                    history.append({"kind": kind, "valid": False, "scored": False,
                                    "reason": "no_slack"})
                    queries += 1
                    continue
                pad_len = int(rng.integers(1, slack_cap + 1))
                try:
                    cand = apply_section_slack_pad(
                        pe_bytes,
                        self._supplement_or_random_payload(rng, pad_len),
                        section_index=-1,
                    )
                except ValueError:
                    history.append({"kind": kind, "valid": False, "scored": False,
                                    "reason": "slack_apply_failed"})
                    queries += 1
                    continue
            elif mode == "combined_multi_region":
                kind = "combined_multi_region"
                self.n_combined_attempts += 1
                # Payload sizes: use the full editable capacity for each
                # region so every candidate exercises the whole envelope.
                p_dos = self._supplement_or_random_payload(rng, max(capacity, 1))
                p_slack = self._supplement_or_random_payload(rng, max(slack_cap, 1))
                p_over = self._supplement_or_random_payload(rng, self.payload_size)
                try:
                    combined = apply_combined_multi_region(
                        pe_bytes, p_dos, p_slack, p_over, section_index=-1,
                    )
                except ValueError:
                    history.append({"kind": kind, "valid": False, "scored": False,
                                    "reason": "combined_apply_failed"})
                    queries += 1
                    continue
                required = {"fulldos", "section_slack", "overlay"}
                if not required.issubset(combined.regions):
                    history.append({
                        "kind": kind,
                        "valid": False,
                        "scored": False,
                        "reason": "combined_regions_incomplete",
                        "regions": list(combined.regions),
                    })
                    queries += 1
                    continue
                cand = combined.data
            else:
                kind = "gamma_padding" if self.benign_payloads else "padding"
                cand = apply_padding(pe_bytes, self._supplement_or_random_payload(rng, self.payload_size))
            queries += 1
            gate = evaluate_transform_validity(pe_bytes, cand, kind=kind)
            # C6 — capability-preservation gate: any candidate that drops a
            # capability from the baseline is rejected before we score it.
            # The gate is a no-op unless capa_preserve was requested.
            capa_ok = self._capa_preserved(capa_baseline, cand)
            extracted = extract_ember_features(cand, extractor=self.extractor)
            # E12 — pefilewarnings-quiet gate: reject candidates whose
            # pefilewarnings band differs from the clean baseline.
            pefw_ok = True
            if (self.fulldos_quiet_only and self._pefw_baseline is not None
                    and extracted.get("features") is not None):
                try:
                    from ..malware.ember2024_extract import pefilewarnings_offset_dim
                    off, dim = pefilewarnings_offset_dim()
                    band = np.asarray(extracted["features"],
                                      dtype=np.float32)[off : off + dim]
                    if not np.array_equal(band, self._pefw_baseline):
                        pefw_ok = False
                        self.n_pefw_rejects += 1
                except Exception:
                    pass
            if extracted.get("features") is None or not capa_ok or not pefw_ok:
                history.append({
                    "kind": kind,
                    "valid": bool(gate.get("passed")),
                    "scored": False,
                    "capa_preserved": bool(capa_ok),
                    "pefw_quiet": bool(pefw_ok),
                })
                continue
            pred, p_mal = predict_malware(self.model, extracted["features"])
            flipped = bool(pred != int(y))
            valid = bool(gate.get("passed"))
            history.append({
                "kind": kind, "valid": valid, "p_malware": p_mal, "flipped": flipped,
                "capa_preserved": True,
            })
            if valid and p_mal < best_p:
                best_p = p_mal
                best_kind = kind
                best_valid = True
                best_bytes = cand
            if valid and flipped:
                best_bytes = cand  # flipped candidate is always the "best"
                return {
                    "chosen_attack": kind,
                    "selected_attack_impl": self.selected_attack_impl,
                    "space": "problem",
                    "success": True,
                    "realizable": True,
                    "scored": True,
                    "best_bytes": best_bytes,   # D9 — persistable
                    "queries_used": queries,
                    "clean_p_malware": p0,
                    "best_p_malware": p_mal,
                    "functionality": gate,
                    "gamma": bool(self.benign_payloads and kind == "gamma_padding"),
                    "secml_gamma_available": try_import_secml_gamma() is not None,
                }
            if queries >= self.n_queries:
                break

        return {
            "chosen_attack": best_kind,
            "selected_attack_impl": self.selected_attack_impl,
            "space": "problem",
            "success": False,
            "realizable": bool(best_valid),
            "scored": True,
            "queries_used": queries,
            "clean_p_malware": p0,
            "best_p_malware": best_p,
            "gamma": bool(self.benign_payloads),
            "secml_gamma_available": try_import_secml_gamma() is not None,
            "attempts": len(history),
            "best_bytes": best_bytes,   # D9 — persistable
            "capa_preserve": bool(self.capa_preserve),
            "capa_preserve_mode": self.capa_preserve_mode if self.capa_preserve else None,
            "capa_baseline_available": bool(self._capa_ok) if self.capa_preserve else None,
            "capa_baseline_error": self._capa_error if self.capa_preserve else None,
            "n_capa_calls": int(self.n_capa_calls) if self.capa_preserve else 0,
            "n_capa_rejects": int(self.n_capa_rejects) if self.capa_preserve else 0,
            "enable_section_slack": bool(self.enable_section_slack),
            "section_slack_capacity": int(slack_cap),
            "n_section_slack_attempts": int(self.n_section_slack_attempts),
            "n_section_slack_no_capacity": int(self.n_section_slack_no_capacity),
            "n_supplement_payloads": int(len(self.supplement_payloads)),
            "transform_set": self.transform_set,
            "n_combined_attempts": int(self.n_combined_attempts),
            "fulldos_quiet_only": bool(self.fulldos_quiet_only),
            "n_pefw_rejects": int(self.n_pefw_rejects),
        }
