"""
`neurinspectre doctor` - lightweight environment sanity checks.

This command is intentionally conservative: it does not attempt network access or
download assets. It helps artifact evaluators confirm that optional dependencies
are installed and that local model metadata is present.
"""

from __future__ import annotations

import hashlib
import re
import sys
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click

from .metadata import collect_env_metadata, collect_git_metadata


def _scan_stub_models(models_dir: Path) -> Tuple[int, List[str]]:
    """
    Scan `models_dir` for `.meta.json` files that mark `is_stub=true`.
    """

    stubbed: List[str] = []
    try:
        for meta_path in sorted(models_dir.glob("*.meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if bool(meta.get("is_stub", False)) or bool(meta.get("stub", False)) or bool(meta.get("placeholder", False)):
                stubbed.append(meta_path.name)
    except Exception:
        return 0, []
    return len(stubbed), stubbed


def _find_pyproject(start: Path) -> Optional[Path]:
    cur = start.resolve()
    for _ in range(20):
        cand = cur / "pyproject.toml"
        if cand.exists():
            return cand
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _parse_toml_string_list(lines: Iterable[str]) -> List[str]:
    """
    Parse a TOML list-of-strings body (best-effort).

    This intentionally avoids adding a hard dependency on `tomli` for py310.
    """

    out: List[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        # Expect: "pkg>=1.2.3",
        m = re.search(r'"([^"]+)"', ln)
        if m:
            out.append(m.group(1).strip())
    return out


def _read_declared_dependencies(pyproject_path: Path) -> Dict[str, List[str]]:
    """
    Return declared dependencies from `pyproject.toml` (project + optional extras).
    """

    text = pyproject_path.read_text(encoding="utf-8", errors="replace").splitlines()

    def _section_slice(header: str) -> Tuple[int, int]:
        start = None
        for i, ln in enumerate(text):
            if ln.strip() == header:
                start = i + 1
                break
        if start is None:
            return -1, -1
        end = len(text)
        for j in range(start, len(text)):
            if text[j].strip().startswith("[") and text[j].strip().endswith("]"):
                end = j
                break
        return start, end

    deps: Dict[str, List[str]] = {"dependencies": [], "optional": []}

    # [project] dependencies = [...]
    s, e = _section_slice("[project]")
    if s != -1:
        i = s
        while i < e:
            ln = text[i]
            if re.match(r"^\s*dependencies\s*=\s*\[\s*$", ln):
                body: List[str] = []
                i += 1
                while i < e and "]" not in text[i]:
                    body.append(text[i])
                    i += 1
                deps["dependencies"] = _parse_toml_string_list(body)
                break
            i += 1

    # [project.optional-dependencies]
    s2, e2 = _section_slice("[project.optional-dependencies]")
    if s2 != -1:
        i = s2
        while i < e2:
            ln = text[i]
            m = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*=\s*\[\s*$", ln)
            if not m:
                i += 1
                continue
            body: List[str] = []
            i += 1
            while i < e2 and "]" not in text[i]:
                body.append(text[i])
                i += 1
            deps["optional"].extend(_parse_toml_string_list(body))
            i += 1

    return deps


def _dist_name_from_requirement(req: str) -> Optional[str]:
    r = str(req or "").strip()
    if not r:
        return None

    # PEP508 direct URL: "name @ url"
    if "@" in r:
        left = r.split("@", 1)[0].strip()
        # Strip extras if any: "name[extra]"
        left = left.split("[", 1)[0].strip()
        return left or None

    # Common: "name>=1.2.3" / "name==..." / "name; marker"
    # Strip environment markers
    r = r.split(";", 1)[0].strip()
    # Strip extras
    r = r.split("[", 1)[0].strip()
    # Stop at first version/operator/space
    m = re.match(r"^([A-Za-z0-9_.-]+)", r)
    return m.group(1) if m else None


def _installed_version(dist_name: str) -> Optional[str]:
    try:
        return str(importlib_metadata.version(dist_name))
    except Exception:
        return None


def _sha256_installed_neurinspectre() -> Dict[str, Any]:
    """
    Best-effort SHA256 of installed `neurinspectre` Python sources.

    We hash `.py` files only to keep the operation bounded and deterministic.
    """

    try:
        import neurinspectre as pkg  # type: ignore
    except Exception as exc:
        return {"available": False, "reason": f"import_failed: {exc}"}

    pkg_dir = Path(getattr(pkg, "__file__", "")).resolve().parent
    if not pkg_dir.exists():
        return {"available": False, "reason": "package_dir_missing", "package_dir": str(pkg_dir)}

    h = hashlib.sha256()
    files: List[Path] = []
    try:
        for p in pkg_dir.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            if not p.is_file():
                continue
            files.append(p)
    except Exception as exc:
        return {"available": False, "reason": f"walk_failed: {exc}", "package_dir": str(pkg_dir)}

    files.sort(key=lambda p: str(p))
    total_bytes = 0
    total_lines = 0
    for p in files:
        rel = str(p.relative_to(pkg_dir)).encode("utf-8", errors="ignore")
        h.update(rel + b"\n")
        try:
            data = p.read_bytes()
        except Exception:
            data = b""
        total_bytes += len(data)
        # Count lines without allocating a large split list.
        if data:
            total_lines += int(data.count(b"\n")) + (0 if data.endswith(b"\n") else 1)
        h.update(data)

    return {
        "available": True,
        "package_dir": str(pkg_dir),
        "file_count": int(len(files)),
        "bytes_hashed": int(total_bytes),
        "py_lines": int(total_lines),
        "sha256": h.hexdigest(),
    }


def _torch_device_report() -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "reason": f"torch_import_failed: {exc}"}

    rep: Dict[str, Any] = {
        "available": True,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    if rep["cuda_available"]:
        try:
            rep["gpu_name"] = str(torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            rep["vram_bytes"] = int(getattr(props, "total_memory", 0))
            rep["vram_gb"] = float(rep["vram_bytes"]) / 1e9 if rep["vram_bytes"] else None
        except Exception:
            pass
    return rep


def _audit_readiness() -> Dict[str, Any]:
    """Local assets needed by `neurinspectre audit`. No network."""
    from neurinspectre.malware.ember_extract import extractor_status
    from neurinspectre.malware.ember2024_extract import extractor_status as ember2024_extractor_status

    carmon = Path("models/cifar10/Linf/Carmon2019Unlabeled.pt")
    gbdt = Path("data/ember/ember2018/ember_model_2018.txt")
    ember_vectors = Path("data/ember/ember_2018/X_test.dat")
    ext = extractor_status()
    gbdt2024 = Path("data/ember/ember2024/EMBER2024_PE.model")
    gbdt2024_win32 = Path("data/ember/ember2024/EMBER2024_Win32.model")
    gbdt2024_win64 = Path("data/ember/ember2024/EMBER2024_Win64.model")
    gbdt2024_apk = Path("data/ember/ember2024/EMBER2024_APK.model")
    gbdt2024_elf = Path("data/ember/ember2024/EMBER2024_ELF.model")
    gbdt2024_pdf = Path("data/ember/ember2024/EMBER2024_PDF.model")
    gbdt2024_dotnet = Path("data/ember/ember2024/EMBER2024_Dot_Net.model")
    gbdt2024_all = Path("data/ember/ember2024/EMBER2024_all.model")
    ext2024 = ember2024_extractor_status()
    return {
        "cli": (
            "neurinspectre audit --target "
            "{carmon,jpeg-carmon,ember-gbdt,ember2024-gbdt,"
            "ember2024-win32-gbdt,ember2024-win64-gbdt,"
            "ember2024-apk-gbdt,ember2024-elf-gbdt,ember2024-pdf-gbdt,"
            "ember2024-dotnet-gbdt,ember2024-all-gbdt}"
        ),
        "carmon_checkpoint": {"path": str(carmon), "present": carmon.is_file()},
        "ember_gbdt": {"path": str(gbdt), "present": gbdt.is_file()},
        "ember_feature_memmap": {"path": str(ember_vectors), "present": ember_vectors.is_file()},
        "ember_extractor": ext,
        "ember2024_gbdt": {"path": str(gbdt2024), "present": gbdt2024.is_file()},
        "ember2024_win32_gbdt": {"path": str(gbdt2024_win32), "present": gbdt2024_win32.is_file()},
        "ember2024_win64_gbdt": {"path": str(gbdt2024_win64), "present": gbdt2024_win64.is_file()},
        "ember2024_apk_gbdt": {"path": str(gbdt2024_apk), "present": gbdt2024_apk.is_file()},
        "ember2024_elf_gbdt": {"path": str(gbdt2024_elf), "present": gbdt2024_elf.is_file()},
        "ember2024_pdf_gbdt": {"path": str(gbdt2024_pdf), "present": gbdt2024_pdf.is_file()},
        "ember2024_dotnet_gbdt": {"path": str(gbdt2024_dotnet), "present": gbdt2024_dotnet.is_file()},
        "ember2024_all_gbdt": {"path": str(gbdt2024_all), "present": gbdt2024_all.is_file()},
        "ember2024_extractor": ext2024,
        "same_sample_note": (
            "EMBER2018 has no PE binaries. Same-sample Full DOS / padding "
            "requires --pe-sample. Feature-space ASR is not PE-valid."
        ),
        "measurement_frame_cli": {
            "scope_pe_corpus": (
                "neurinspectre scope-pe-corpus <pe_dir> "
                "--supplement-index data/ember/ember2024/capa_supplement_index.json"
            ),
            "audit_with_crossing": (
                "neurinspectre audit --target ember2024-gbdt --pe-sample <dir> "
                "--save-best-bytes --crossing-matrix --write-diagnosis"
            ),
            "transferability": (
                "neurinspectre transferability <audit_dir>/audit_report.json --default-crossing"
            ),
            "capa_diff": "neurinspectre capa-diff-audit <audit_dir>/audit_report.json",
            "diagnose": "neurinspectre diagnose-ember-audit <audit_dir>",
            "pipeline_info": "neurinspectre ember-pipeline-info --target ember2024-gbdt",
        },
    }


def run_doctor(ctx: click.Context, **kwargs: Any) -> None:
    json_output = kwargs.get("json_output")
    as_json = bool(kwargs.get("as_json", False))
    models_dir = Path(str(kwargs.get("models_dir", "models")))
    check_models = bool(kwargs.get("check_models", True))

    payload: Dict[str, Any] = {
        "env": collect_env_metadata(),
        "git": collect_git_metadata(),
    }
    payload["torch_device"] = _torch_device_report()
    payload["package_hash"] = _sha256_installed_neurinspectre()
    payload["audit"] = _audit_readiness()

    # Dependency inventory: declared (pyproject) -> installed versions.
    deps: Dict[str, Any] = {"declared": None, "installed": {}, "missing": []}
    pyproject = _find_pyproject(Path.cwd())
    if pyproject:
        declared = _read_declared_dependencies(pyproject)
        deps["declared"] = {"pyproject_path": str(pyproject), **declared}
        names: List[str] = []
        for req in list(declared.get("dependencies") or []) + list(declared.get("optional") or []):
            dn = _dist_name_from_requirement(req)
            if dn and dn not in names:
                names.append(dn)
        installed: Dict[str, Optional[str]] = {}
        missing: List[str] = []
        for dn in names:
            v = _installed_version(dn)
            installed[dn] = v
            if v is None:
                missing.append(dn)
        deps["installed"] = installed
        deps["missing"] = missing
    payload["dependencies"] = deps

    if check_models and models_dir.exists() and models_dir.is_dir():
        stub_count, stub_names = _scan_stub_models(models_dir)
        payload["models"] = {
            "models_dir": str(models_dir),
            "stub_meta_count": int(stub_count),
            "stub_meta_files": list(stub_names),
        }
    else:
        payload["models"] = {
            "models_dir": str(models_dir),
            "stub_meta_count": None,
            "stub_meta_files": [],
            "note": "models directory not found or model scanning disabled",
        }

    if json_output:
        out_p = Path(str(json_output))
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        click.echo(f"Wrote {out_p}")

    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    # Human-readable minimal output
    py = (payload.get("env") or {}).get("python") or {}
    plat = (payload.get("env") or {}).get("platform") or {}
    torch = (payload.get("env") or {}).get("torch") or {}
    opt = (payload.get("env") or {}).get("optional_modules") or {}
    torch_dev = payload.get("torch_device") or {}
    git = payload.get("git") or {}
    models = payload.get("models") or {}
    deps = payload.get("dependencies") or {}
    pkg_hash = payload.get("package_hash") or {}

    try:
        pkg_version = importlib_metadata.version("neurinspectre")
    except Exception:
        pkg_version = None

    click.echo(f"Python: {sys.version.strip()} ({py.get('executable')})")
    click.echo(f"Platform: {plat.get('system')} {plat.get('release')} {plat.get('machine')}")
    if pkg_version:
        click.echo(f"NeurInSpectre: {pkg_version}")
    if isinstance(torch, dict) and torch.get("available") is False:
        click.echo("Torch: not available")
    else:
        click.echo(f"PyTorch: {torch.get('version')}")
        cuda_ver = torch_dev.get("cuda_version") if isinstance(torch_dev, dict) else None
        click.echo(f"CUDA: {cuda_ver or 'N/A'}")
        click.echo(f"MPS: {bool(torch.get('mps_available'))}")
        if isinstance(torch_dev, dict) and torch_dev.get("cuda_available"):
            gpu = torch_dev.get("gpu_name")
            vram = torch_dev.get("vram_gb")
            if gpu:
                click.echo(f"GPU: {gpu}")
            if vram is not None:
                click.echo(f"VRAM: {float(vram):.1f} GB")
    click.echo(
        "Optional deps: "
        f"ember={opt.get('ember')} lief={opt.get('lief')} "
        f"lightgbm={opt.get('lightgbm')} pefile={opt.get('pefile')} "
        f"nuscenes={opt.get('nuscenes')} onnxruntime={opt.get('onnxruntime')}"
    )

    if isinstance(git, dict) and git.get("available"):
        click.echo(f"Git: commit={git.get('commit')} dirty={git.get('dirty')} branch={git.get('branch')}")
    else:
        click.echo("Git: not a repo (or git unavailable)")

    click.echo(f"Models: dir={models.get('models_dir')} stub_meta_count={models.get('stub_meta_count')}")
    audit = payload.get("audit") or {}
    if audit:
        carmon = audit.get("carmon_checkpoint") or {}
        gbdt = audit.get("ember_gbdt") or {}
        memmap = audit.get("ember_feature_memmap") or {}
        ext = audit.get("ember_extractor") or {}
        gbdt2024 = audit.get("ember2024_gbdt") or {}
        gbdt2024_win32 = audit.get("ember2024_win32_gbdt") or {}
        gbdt2024_win64 = audit.get("ember2024_win64_gbdt") or {}
        gbdt2024_apk = audit.get("ember2024_apk_gbdt") or {}
        gbdt2024_elf = audit.get("ember2024_elf_gbdt") or {}
        gbdt2024_pdf = audit.get("ember2024_pdf_gbdt") or {}
        gbdt2024_dotnet = audit.get("ember2024_dotnet_gbdt") or {}
        gbdt2024_all = audit.get("ember2024_all_gbdt") or {}
        ext2024 = audit.get("ember2024_extractor") or {}
        click.echo(
            "Audit: "
            f"carmon_ckpt={carmon.get('present')} "
            f"ember_gbdt={gbdt.get('present')} "
            f"ember_memmap={memmap.get('present')} "
            f"extractor={ext.get('available')} "
            f"official_reproduction={ext.get('official_reproduction')}"
        )
        if ext and not ext.get("available", True):
            click.echo(f"Audit extractor: {ext.get('reasons')} — {ext.get('hint') or ''}")
        elif ext.get("mac_inconsistent"):
            click.echo(f"Audit extractor: {ext.get('mac_note')}")
        click.echo(
            "EMBER2024: "
            f"pe={gbdt2024.get('present')} "
            f"win32={gbdt2024_win32.get('present')} "
            f"win64={gbdt2024_win64.get('present')} "
            f"apk={gbdt2024_apk.get('present')} "
            f"elf={gbdt2024_elf.get('present')} "
            f"pdf={gbdt2024_pdf.get('present')} "
            f"dotnet={gbdt2024_dotnet.get('present')} "
            f"all={gbdt2024_all.get('present')} "
            f"extractor={ext2024.get('available')} "
            f"thrember={ext2024.get('thrember_version')} "
            f"pefile={ext2024.get('pefile_version')} "
            f"official_reproduction={ext2024.get('official_reproduction')}"
        )
        if ext2024 and not ext2024.get("available", True):
            click.echo(
                f"EMBER2024 extractor: {ext2024.get('reasons')} — {ext2024.get('hint') or ''}"
            )
        click.echo(f"Audit CLI: {audit.get('cli')}")
        frame = audit.get("measurement_frame_cli") or {}
        if frame:
            click.echo("Measurement frame (PE red-team):")
            for key in (
                "scope_pe_corpus",
                "audit_with_crossing",
                "transferability",
                "capa_diff",
                "diagnose",
                "pipeline_info",
            ):
                if frame.get(key):
                    click.echo(f"  {key}: {frame[key]}")

    if isinstance(pkg_hash, dict) and pkg_hash.get("available"):
        click.echo(f"Installed package sha256 (py): {pkg_hash.get('sha256')}")
        if pkg_hash.get("py_lines") is not None:
            click.echo(f"Installed package LOC (py): {int(pkg_hash.get('py_lines') or 0)}")

    # Print declared dependency versions (AE-friendly; long but explicit).
    declared = (deps.get("declared") or {}) if isinstance(deps, dict) else {}
    installed = (deps.get("installed") or {}) if isinstance(deps, dict) else {}
    if declared and installed:
        click.echo("Dependencies (pyproject.toml -> installed):")
        for req in list(declared.get("dependencies") or []):
            dn = _dist_name_from_requirement(str(req))
            v = installed.get(dn) if dn else None
            click.echo(f"  - {req}  # installed: {v if v is not None else 'MISSING'}")
        if declared.get("optional"):
            click.echo("Optional dependencies (pyproject.toml -> installed):")
            for req in list(declared.get("optional") or []):
                dn = _dist_name_from_requirement(str(req))
                v = installed.get(dn) if dn else None
                click.echo(f"  - {req}  # installed: {v if v is not None else 'MISSING'}")

