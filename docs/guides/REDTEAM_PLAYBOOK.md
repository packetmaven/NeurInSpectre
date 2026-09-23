# Red-team playbook: offensive NeurInSpectre in authorized engagements

> **Tracked copy** (repo: `docs/guides/REDTEAM_PLAYBOOK.md`). A working copy with frozen
> run artifacts may also live at `results/offensive_overnight_20260920/REDTEAM_PLAYBOOK.md`
> (gitignored). EMBER-only quickstart: [AUTHORIZED_REDTEAM_EMBER.md](AUTHORIZED_REDTEAM_EMBER.md).

This is how the offensive stack is used on a **written ROE** against a client ML system. It is not a product pitch and it is not “we break production AV.”

Entry point: `venv/bin/neurinspectre` (Python 3.10). Stale Homebrew 3.11 has no `audit`.

Companion evidence (often under `results/offensive_overnight_20260920/`): `CLAIM_LEDGER.md`, `doctor.json`, `inventory.json`, `ember_5k_diagnosis.json`.

## Engagement model

```
authorized scope → access model → characterize → chosen_attack → attack/audit → validity gate → client report
```

MITRE ATLAS labels that already appear in-repo (offline STIX via `neurinspectre mitre-atlas coverage`):

- **AML.T0043** Craft Adversarial Data — vision Square / PGD / BPDA / NI column
- **AML.T0020** Poison Training Data — Table 5 last-Linear hijack
- **AML.T0024.*** Exfiltration / inversion — Table 4 / `gradient-inversion recover` (prototype only)

Export a coverage appendix on day 1:

```bash
venv/bin/neurinspectre mitre-atlas coverage --scope all --format markdown --out atlas_appendix.md
```

## Day 0 — kickoff (non-negotiable)

Write down, signed:

1. Systems in scope (checkpoint paths, defense wrappers, whether PE bytes may be mutated).
2. Access: whitebox weights, scores, labels, feature vectors, or problem-space files.
3. Query / wall-clock budget.
4. Validity definition the client accepts (pixel L∞, PE parse/structure, **not** “looks fine to a human” unless they define it).
5. No live-malware download. Client supplies binaries if malware scoring is in scope.
6. What must **not** be claimed: official EMBER2018 unless Elastic-verified (non-Darwin, lief 0.9.0 or 0.10.1); FeatureSquare as PE-valid; design-target paper headlines; sandbox/AV success from a GBDT audit alone (see `measurement_scope.not_measured` on the report JSON).

```bash
venv/bin/neurinspectre doctor --as-json --json-output doctor.json
```

Tonight’s doctor: Carmon ckpt present, official GBDT present, EMBER memmap present, extractor available, **`official_reproduction=false`** on Darwin/lief 0.13.2, MPS on, CUDA off.

## Engagement 1 — vision defense as a pipeline (highest confidence)

**When:** client ships JPEG (tonight’s measured case) or a similar **non-differentiable / shattered-gradient** wrapper. Bit-depth / smoothing / pad-crop are in the prior Table 8 artifact, not in tonight’s live reruns.

**Operator loop**

```bash
venv/bin/neurinspectre characterize \
  -m models/cifar10_resnet20_norm_ts.pt -d cifar10 \
  --defense jpeg --krylov-order 20 --num-samples 100 \
  -o characterize_jpeg.json

venv/bin/neurinspectre attack \
  -m models/cifar10_resnet20_norm_ts.pt -d cifar10 \
  --defense jpeg --attack-type neurinspectre --num-samples 100

# RobustBench Carmon + JPEG as a security pipeline
venv/bin/neurinspectre audit --target jpeg-carmon --smoke -o results/audit/jpeg_carmon_smoke
venv/bin/neurinspectre audit --target jpeg-carmon --mode all --query-budgets 100,500,2000,5000
```

**What “success” means:** query-budget ASR **after** the validity gate (clean accuracy not collapsed; perturbation inside the stated norm), compared to PGD and AutoAttack on the **same** samples.

**Tonight’s evidence**

- JPEG table2 slice (folder `table8_jpeg_resnet20/` — freeze name was `table8_jpeg_carmon/`, renamed 22 Sep 2026; still **not Carmon**; ResNet-20 + JPEG, 128 submitted / 101 clean-correct, validity passed, clean 78.9%): PGD **9.9%** (10/101), AutoAttack **98.0%** (99/101), NeurInSpectre **99.0%** (100/101, chosen BPDA). Same *shape* as the paper JPEG row, not the same n/model-budget cell.
- Table 2 smoke, 256 CIFAR-10 test images, JPEG q=75, validity **passed** (clean 80.1%):
  - Characterization selected **BPDA** (shattered + vanishing; `requires_bpda: true`).
  - NeurInSpectre (routed) conditional ASR **98.54%** vs PGD **7.32%**.
- Direct `attack` on the same ResNet-20+JPEG (CPU): NI **96.3%** (26/27, routed BPDA), explicit BPDA **100%** (27/27), Square **91.7%** (11/12, mean ~3500 queries — not a 100-query budget).
- AutoAttack 99/101 on the JPEG slice is Square-dominated (inner APGD-CE/DLR/T flip 3/3/2). Not a gradient win.
- `characterize` (100 samples): shattered+vanishing, `requires_bpda`/`requires_mapgd`, confidence 47.5%. `chosen_attack` is null in that JSON; do not mix with the 128-slice confidence (0.10).
- Carmon/JPEG-Carmon **smokes** (n=8): routing works (jpeg→bpda, carmon→apgd); ASRs 0–14% — not attack-strength evidence.
- Prior paper-grade Table 8 (do not pretend tonight re-ran it): +17.0 pp vs AutoAttack on 8/12 validity-passed defenses.

**JPEG fact the operator must not “fix”:** at quality ≥ 75 the differentiable JPEG path is **identity** (Athalye). BPDA through the true defense forward is the attack, not a fake JPEG.

**Do not tell the client:** “we broke Carmon RobustBench SOTA” unless a full `audit --target jpeg-carmon` (not `--smoke`) finished under their budget.

## Engagement 2 — malware ML score vs file (honest null is the finding)

**When:** client has a static PE scorer (EMBER-like GBDT / 2381-d or EMBER 2024 / 2568-d). Public EMBER releases **have no PE binaries**. Same-sample work requires `--pe-sample`.

### Two frames (write this in the SOW closeout)

| Frame | Question the client often asks | What NeurInSpectre measures |
|---|---|---|
| **Engagement** | Does the file still behave, and did the **deployed** detector miss it? | **Not** this CLI (needs sandbox / EDR / VT — out of scope). |
| **NeurInSpectre** | On this **named checkpoint**, these **byte regions**, under a **parse gate**, at this **query budget**, did the score cross 0.5? | **Yes** — `measurement_scope` on every EMBER audit report lists what is *not* claimed (section injection, IAT rewrites, graph/byte models, sandbox execution, commercial AV). |

Both numbers can be honest. Only the second is what `neurinspectre audit` produces. A score drop without a flip is **not** an evasion finding.

GBDT characterize / `audit.pipeline` is enriched so operators are **not** routed to JPEG BPDA or whitebox PGD (`gradient_available: false`, `recommended_recipe: problem_space`).

**Operator loop (2026-09-23 — validated on 148-file reference corpus)**

```bash
# Day 0 preflight: SHA inventory + challenge / Capa supplement overlap (no attack spend)
venv/bin/neurinspectre scope-pe-corpus /client/malware_pe \
  --challenge-dir data/ember/ember2024/dataset/challenge \
  --supplement-index data/ember/ember2024/capa_supplement_index.json \
  -o results/pe_scope.json

# File-level Capa sidecar for cohort filters (see post-overnight block below)
venv/bin/neurinspectre run-capa /client/malware_pe --sidecar pe_tags.json \
  --supplement-index data/ember/ember2024/capa_supplement_index.json

# EMBER 2024 same-sample audit + client report bundle
venv/bin/neurinspectre audit --target ember2024-gbdt \
  --pe-sample /client/malware_pe \
  --require-detected \
  --query-budgets 10,50,100,500,5000 \
  --save-best-bytes --crossing-matrix \
  -o results/audit/ember2024_client

# Optional: one-shot Capa diff original vs best_bytes (file_level ~seconds/file; full ~minutes/file)
venv/bin/neurinspectre audit --target ember2024-gbdt \
  --pe-sample /client/malware_pe --n-examples 4 \
  --require-detected --save-best-bytes \
  --capa-diff-best --capa-diff-backend file_level \
  -o results/audit/ember2024_capa_diff_smoke

# One-shot operator bundle: scope + vt_sidecar + audit + diagnosis + crossing + zip
# (save-best-bytes and crossing on by default; pass --sow-adapter* to run adapters after audit)
venv/bin/neurinspectre redteam-bundle /client/malware_pe \
  -o results/audit/ember2024_client_bundle \
  --target ember2024-gbdt --require-detected \
  --query-budgets 10,50,100,500,5000

# Read-only VT-style metadata (challenge JSONL lookup by SHA; no live submit)
venv/bin/neurinspectre vt-sidecar /client/malware_pe -o results/vt_sidecar.json
venv/bin/neurinspectre audit --target ember2024-gbdt --pe-sample /client/malware_pe \
  --vt-sidecar results/vt_sidecar.json --write-diagnosis ...

# Bounded IAT edits (default: API name case toggle; DLL case is usually a thrember no-op)
venv/bin/neurinspectre iat-probe /client/malware_pe/sample.exe
venv/bin/neurinspectre audit --target ember2024-gbdt --pe-sample /client/malware_pe \
  --enable-iat-edits --require-detected ...

# SOW adapters (export only — no in-process sandbox or AV API)
venv/bin/neurinspectre audit ... --save-best-bytes \
  --sow-adapter sandbox_handoff
# commercial_av_edr placeholder requires --sow-adapter-ack and --av-system-name

# Compact ledger cell for slides / SOW appendix
python scripts/diagnose_ember_audit.py results/audit/ember2024_client

# Offline crossing matrix (same rule as transferability: clean ≥0.5, bytes changed, mutated <0.5)
venv/bin/neurinspectre transferability results/audit/ember2024_client/audit_report.json \
  --default-crossing -o results/audit/ember2024_client/crossing_matrix_cli.json
```

`--crossing-matrix` and `--capa-diff-best` **require** `--save-best-bytes` (fail fast at CLI start).

**GAMMA section injection (secml-malware; separate venv from EMBER2018 lief 0.9.0)**

```bash
pip install -e '.[gamma]'
neurinspectre doctor   # smoke inject + donor preflight
neurinspectre gamma-inject /client/malware_pe/sample.exe --gamma-donor-dir /client/benign_pe
neurinspectre audit --target ember2024-gbdt --pe-sample /client/malware_pe \
  --enable-gamma-sections --gamma-donor-dir /client/benign_pe \
  --require-detected --query-budgets 10,50,100,500,5000 -o results/audit/ember2024_gamma
```

Use a **real benign PE donor corpus** (not pip `goodware_samples` text stubs). With `--enable-gamma-sections`, `measurement_scope` records GAMMA as measured; overlay `--benign-corpus` padding is still a different transform.

**EMBER 2018 official cell (Elastic reproduction, not Mac `audit`)**

```bash
# Fail closed unless Elastic-verified extractor on the audit host
venv/bin/neurinspectre audit --target ember-gbdt \
  --pe-sample /client/malware_pe --require-official-reproduction

# Quote_as_ember2018 numbers: LIEF 0.9.0 Linux harness only
bash scripts/audit_ember_linux_lief090.sh /client/malware_pe
# Finished reference: results/audit/ember_gbdt_linux_lief090/ — n=58, 500q, PE-valid ASR 0.0
```

Linux helpers: `bash scripts/audit_ember_linux.sh /pe_dir` or `bash scripts/audit_ember_linux_native.sh /pe_dir`.

**Two numbers, never one (feature vs PE-valid)**

| Object | Attack | Realizable? | Reference corpus |
|---|---|---|---|
| Feature vector | FeatureSquare, L∞ ε=1.0 | **No** (mixed-scale) | August unofficial: ASR **1.0** (56 PEs, lief 0.13.2, 5000q) |
| PE bytes | Full DOS + overlay (+ optional section-slack) | Parse-valid only | Official 2018: **0/58** @ 500q (lief 0.9.0); August: **0/56** @ 5000q |

On the reference 148-file corpus, a recent smoke with `--query-budgets 10,50` on 8 detected 2024 samples: problem-valid ASR **0.0**, max drop **~0.064**, closest best-p still **≥0.5** (still malware under the named model). Crossing matrix on EMBER2018 + PE + Win32 + Win64: **0/8** transfers — expected on this corpus.

**Client takeaway:** feature-space ASR is not a PE-valid red-team finding. A parse-valid ASR of zero with documented closest approach is a defensible result. Do not conflate with “Defender missed it” unless the SOW includes that system.

**Engagement gaps:** `measurement_scope.not_measured` always includes sandbox execution, commercial AV/EDR, and graph/byte models unless the SOW uses separate tooling outside this CLI. Optional audit flags move **bounded** transforms into `measured` when enabled: GAMMA section injection (`--enable-gamma-sections`), import-table edits (`--enable-iat-edits`; default search uses API name case toggles and skips thrember no-ops — use `iat-probe` first). SOW adapters (`--sow-adapter sandbox_handoff` or `commercial_av_edr` with ack) add export/provenance under `measured` but **do not** remove sandbox or AV from `not_measured`. VT ratios come from a **read-only** sidecar (`vt-sidecar` / `--vt-sidecar`), not live VirusTotal. Operator catalog:

```bash
neurinspectre engagement-gaps
```

PyRIT attack-planning remains out of scope unless the SOW adds it separately.

## Engagement 3 — last-layer supply-chain / fine-tune hijack (medium confidence)

**When:** the engagement includes **training-time** or last-layer fine-tune access (not inference-only).

```bash
python scripts/reproduce_module_table5.py --n-seeds 3 --baseline \
  --epochs 8 --lr 0.05 --poison-rate 0.25 --n-train 4000 --n-test 2000 --nc-iter 80 \
  --output-dir results/table5_engagement
```

**What to report:** BD ASR vs Neural Cleanse anomaly (paper threshold 2.0) and STRIP triggered-flag (paper 85%). Tonight’s production-hparams re-run reproduced BD ASR **24.698%** and STRIP **37.48%**; NC 1.439 vs prior 1.434. Default script flags (3 epochs, poison 0.1, 2000 train) only reach **3.49%**. All three seeds were identical (std=0) — do not report a 3-seed CI. **Do not** quote ResNet-50 94–97% (protocol not shipped).

ATLAS: AML.T0020 poison training data.

## Engagement 4 — design-target modules (demo / lab only)

These CLIs **start**. Paper headlines **do not** reproduce. Put them in the SOW as research prototypes or omit them.

| Tool | Tonight | Why it is not a billed finding |
|---|---|---|
| `statistical-evasion` / `drift-detect` | Detects a planted 0.8 mean shift (consensus drift true) | Synthetic; not 10/12 Fisher |
| `rl-obfuscation analyze` | S_RL 0.386 LOW on random gradients | Weights are heuristic; paper 96.8% not shipped |
| `activation_steganography encode` | 4 bits Hamming-coded into a gpt2 prompt | Not 3.2 bits/neuron |
| `attention-security` | IsolationForest on gpt2 tokens | No 93.4% head-detection metric |
| `adversarial-ednn` | Crashed: embeddings flattened | Missing/broken path |
| `gradient-inversion recover` / Table 4 | Tonight NI SSIM **0.019** unscreened, H_S mean **0.80**, 0/20 passed H_S<0.3 | Paper SSIM 0.89 is design-target |
| `red-team attack-planning` | HTML dashboard from two prompts | Visualization, not an exploit |

## Staffing a 5-day authorized engagement

**Day 1.** `doctor`. Characterize 1–2 production-like defenses. `table2-smoke` to prove wiring + validity gates. ATLAS coverage appendix. Confirm extractor / official-reproduction flags if malware is in scope.

**Days 2–n.** Budgeted `audit` / `attack` on the in-scope access (`--mode whitebox|scores|labels|feature|problem|all`). One GPU job at a time on MPS. Keep a claim ledger like this folder.

**Closeout.** For each cell: object attacked, gate, n, budget, ASR, `official_reproduction`, sellable (yes/no). Partial + honest beats a silent incomplete Table 8.

## Commands that must work (canonical)

```bash
neurinspectre audit --target carmon --smoke
neurinspectre audit --target jpeg-carmon --smoke
neurinspectre audit --target ember-gbdt --smoke
neurinspectre audit --target ember-gbdt --smoke --pe-sample /path/to/pe_or_dir
neurinspectre audit --target ember2024-gbdt --pe-sample ./pe --save-best-bytes --crossing-matrix
neurinspectre scope-pe-corpus ./pe
neurinspectre config audit --target ember-gbdt --smoke --pe-sample ./pe_dir
neurinspectre doctor
```

`--require-official-reproduction` fails closed unless Elastic-verified. `--require-detected` fails if the GBDT kept no malware PEs.

## Post-overnight additions (2026-09-22) — EMBER 2024 lane

Landed after the 20 Sep freeze. Fully rowed in the ledger's *Post-overnight extensions* section. Engagement mapping:

**Engagement 2 (malware ML score vs file)** — operator tools (see full loop above):

```bash
# Preflight SHA / challenge / supplement overlap (no queries burned)
neurinspectre scope-pe-corpus ./pe \
  --challenge-dir data/ember/ember2024/dataset/challenge \
  --supplement-index data/ember/ember2024/capa_supplement_index.json

# File-level Capa + supplement merge when SHA hits the index
neurinspectre run-capa ./pe --sidecar pe_tags.json \
  --supplement-index data/ember/ember2024/capa_supplement_index.json

# Cohort filters (sidecar from run-capa or challenge tags)
neurinspectre audit --target ember2024-gbdt --pe-sample ./pe --n-examples 100 \
  --filter-tags-json pe_tags.json \
  --filter-capability "Encode data using xor" \
  --filter-tag "att&ck:T1055" --filter-tag "mbc:C0059" --filter-family emotet

# Report bundle: best bytes + four-model crossing matrix (2018 + PE/Win32/Win64)
neurinspectre audit --target ember2024-gbdt --pe-sample ./pe \
  --require-detected --save-best-bytes --crossing-matrix \
  --query-budgets 10,50,100,500,5000

# Capa before/after on best_bytes (optional; not a behavior gate)
neurinspectre audit --target ember2024-gbdt --pe-sample ./pe \
  --save-best-bytes --capa-diff-best --capa-diff-backend file_level

# Stricter search gates (unchanged from 22 Sep)
neurinspectre audit --target ember2024-gbdt --pe-sample ./pe \
  --capa-preserve --capa-preserve-mode ttps \
  --enable-section-slack --fulldos-quiet-only

# Offline re-score (or use --default-crossing for the shipped four checkpoints)
neurinspectre transferability <audit_dir>/audit_report.json --default-crossing
neurinspectre transferability <audit_dir>/audit_report.json \
  -m PE=data/ember/ember2024/EMBER2024_PE.model \
  -m Win32=data/ember/ember2024/EMBER2024_Win32.model \
  -m Win64=data/ember/ember2024/EMBER2024_Win64.model

python scripts/diagnose_ember_audit.py <audit_dir>

# AdvMal-TF / PhantomCall byte-envelope label only (not their generators)
neurinspectre audit --target ember2024-gbdt --pe-sample ./pe --transform-set combined
```

Every EMBER audit JSON should carry **`measurement_scope`** and an enriched **`pipeline`** block. Use `diagnose_ember_audit.py` for `query_curve`, `closest`, and `closest_still_malicious` (drop but still ≥0.5).

What these do **and do not** deliver in a client report:

| Capability | What a client can be told | What must **not** be claimed |
|---|---|---|
| `--enable-section-slack` | New transform that preserves file length and every function byte. On the smoke, 2/4 files chose it. The rescaled move is 0.895 → 0.883. | It did not flip any file. |
| `--capa-preserve-mode ttps` | Stricter validity: retains every ATT&CK-tagged capability. | The saved C8 run is n=2, 0 rejects, and the same chosen attacks in all/ttps/mbc. It is not a 4-file result. |
| `--fulldos-quiet-only` | Hard invariant: rejects payloads that perturb the pefilewarnings feature. | Not "fixing a leak". `results/ember2024/E12/fulldos_pefw_probe.json` is 0/400, max L1 0.0. |
| `neurinspectre transferability` / `--crossing-matrix` | Re-scores clean and mutated bytes on **EMBER2018 + PE + Win32 + Win64** when models exist on disk. A flip is a crossing (clean ≥ 0.5, bytes changed, mutated < 0.5). | On saved D9 and on 2026-09-23 smoke (8 samples) crossing rate is **0** — not a multi-model evasion claim. Mac 2018 scores in the matrix are **comparison**, not `quote_as_ember2018`. |
| `scope-pe-corpus` | SHA inventory; challenge + supplement overlap before attack budget. | Supplement overlap **0/148** on the reference corpus is expected; does not block file-level Capa sidecars. |
| `measurement_scope` / `--capa-diff-best` | Documents SOW boundaries; optional Capa diff on original vs best_bytes. | Capa diff is **not** sandbox execution preservation. |
| `--transform-set combined` | Multi-region envelope both AdvMal-TF (May 2026) and PhantomCall (Sep 2026) describe. A candidate is combined only when Full DOS, section-slack, and overlay all change bytes. | The saved smoke (0/4) is the pre-fix envelope. It is not a combined ASR and does not reproduce their 85–98% headlines. |

**Engagement 4 (design-target / research)** gets one item, kept honest:

- `neurinspectre train-function-ml` — opcode n-gram LightGBM over the 23.8 GB Capa supplement. Test AUC **0.984–0.996** on top-20 capabilities, sha256-split held-out. This is a **defender-side finding** (capabilities are learnable from disassembly opcode-shape alone), not an evasion capability. Cite as: "an attacker who wants to hide a capability at the function level must alter opcode-shape, not just call targets or byte-level obfuscation."

**Engagement 2 corpus intel** (no new attack):

- `neurinspectre index-capa-supplement` → 1.4 GB SHA-256 → per-function metadata index (800,470 files × 16,356,790 functions × 583 unique capabilities).
- `neurinspectre lookup-capa-functions <sha256>` — check whether a file in scope has known-capability functions in the EMBER 2024 supplement. The lookup reads that hash only. On our reference user corpus, 0/148 SHAs matched.
- `neurinspectre run-capa ./pe --sidecar pe_tags.json --supplement-index data/ember/ember2024/capa_supplement_index.json` writes the sidecar `audit --filter-tags-json` consumes. Audit rows then carry the file-level Capa names, ATT&CK techniques, MBC behaviors, and whether the supplement had function labels. The saved sidecar for the 148-file corpus is `results/ember2024/capa_filter_sidecar.json`: 0 supplement hits, 88 files match `reference analysis tools strings`.

Everything under this section is fully unit-tested (**265 passed** on 22 Sep 2026 across the 17 offensive-lane modules). Not a workshop; the flags exist and are documented, but the results the operator will get on their own PE corpus require their own SHAs.

## What not to sell

- FeatureSquare as PE-valid.
- Mac or lief 0.13.x GBDT scores as EMBER2018.
- Table 8 EMBER MLP 0% rows as “we broke EMBER.”
- RL 96.8% / EDNN 91.7% / steg 3.2 bits / attention 93.4% / inversion SSIM 0.89.
- Month 4 test-time class / router / ensemble members.
- A workshop where attendees must fight PATH and missing extras. Talk-ready; workshop not ready.
- Section-slack / TTP-preserve / combined-transform / transferability / warnings-quiet as PE-valid evasion on EMBER 2024. Currently all null on reference corpora.
- “Deployed AV missed it” or “still runs in sandbox” inferred from a GBDT `audit_report.json` alone (`measurement_scope.not_measured`).
- Crossing-matrix **0/N** as “we evaded four detectors” without the crossing rule and per-model clean_p.
- E11 function-ML as "we broke Capa." It is a *learnability* result that helps defenders.
- The Capa supplement index as "we have your file's disassembly." SHA lookup only; no supplement entry unless the file is in FutureComputing4AI/EMBER2024.
