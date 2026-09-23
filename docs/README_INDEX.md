# README Index (Complete)

## Authorized red team (EMBER / vision)

- [AUTHORIZED_REDTEAM_EMBER.md](guides/AUTHORIZED_REDTEAM_EMBER.md) — EMBER GBDT same-sample audit, `measurement_scope`, crossing matrix.
- [REDTEAM_PLAYBOOK.md](guides/REDTEAM_PLAYBOOK.md) — full engagement playbook (vision, malware, Table 5, EMBER 2024 lane).

## Operations Quickstart (Verified)

- One-time local setup (recommended):
  ```bash
  mkdir -p _cli_runs viz/gpu
  ```

- TTD Dashboard (Time Travel Debugger) - WITH MODEL SWITCHING
  - Start (SIMPLE SYNTAX - recommended):
    ```bash
    # Prep a small local gradient dataset (writes: _cli_runs/adversarial_obfuscated_gradients.npy)
    neurinspectre obfuscated-gradient capture-adversarial --attack-type combined --output-dir _cli_runs

    # Start dashboard with explicit data inputs (gradient-file takes precedence over batch-dir)
    neurinspectre dashboard \
      --model gpt2 \
      --port 8080 \
      --attention-file real_attention.npy \
      --gradient-file _cli_runs/adversarial_obfuscated_gradients.npy
    ```
  - Or use dashboard manager:
    ```bash
    neurinspectre dashboard-manager start --dashboard ttd
    ```
  - Test connectivity:
    ```bash
    /usr/bin/curl -sS http://127.0.0.1:8080 | head -n 5
    ```
  - Features:
    - 🧠 Model Dropdown: Switch between GPT-2, T5, RoBERTa, BERT, DistilBERT
    - 📊 Real Data: 475MB adversarial gradient dataset with rolling window
    - 🍎 Apple Silicon: MPS optimized for Mac M1/M2/M3
    - 🛡️ MITRE ATLAS: 2024-2025 research-validated threat intelligence
  - Logs (writable by default):
    - Default directory: system temp under `neurinspectre_logs` (override via `NEURINSPECTRE_LOG_DIR`)
    - Example: `/var/folders/.../T/neurinspectre_logs/ttd_dash8080.log`
  - Stop:
    ```bash
    neurinspectre dashboard-manager stop --dashboard ttd
    ```

- GPU Detection (Apple Silicon MPS/CUDA autodetect)
  ```bash
  neurinspectre gpu detect --output viz/gpu/gpu_report.json
  ```

- Spectral Analysis - Interactive Dashboard
  ```bash
  # Generate a small local demo input (writes: _cli_runs/generated_obfuscated_gradients.npy)
  neurinspectre obfuscated-gradient generate --samples 512 --output-dir _cli_runs

  neurinspectre math spectral --input _cli_runs/generated_obfuscated_gradients.npy --output _cli_runs/spectral.json --plot _cli_runs/spectral.png
  ```
  
  **📊 Output Files (saved to `_cli_runs/`):**
  ```bash
  # PRIMARY: Interactive Dashboard (use this!)
  open _cli_runs/spectral_interactive.html
  
  # FALLBACK: Static PNG for reports
  open _cli_runs/spectral.png
  
  # RAW DATA: JSON with all metrics
  open _cli_runs/spectral.json
  
  # List all spectral outputs
  ls -la _cli_runs/spectral*
  ```
  
  **Why the HTML?**
  - ✅ Fully Interactive: Zoom, pan, scroll through large datasets
  - ✅ Persistent Hover: Detailed tooltips stay visible during zoom
  - ✅ Red/Blue Guidance: Every data point has offensive/defensive intel
  - ✅ 4 Panels: Original Signal, Spectral Magnitude, Obfuscation Indicators, Summary Metrics
  - ✅ Color-coded Threats: Orange/red markers for critical findings
  - ✅ MITRE ATLAS Mapping: Each anomaly mapped to attack techniques
  
  **Red Team Usage:**
  - Hover over high gradient points (orange/red) → See MI attack targets
  - Check spectral peaks → Find obfuscation signatures to exploit
  - Review obfuscation indicators → Understand attack surface (MITRE ATLAS)
  
  **Blue Team Usage:**
  - Hover over high gradients → Get clipping thresholds and DP noise values
  - Monitor spectral anomalies → Detect obfuscation patterns early
  - Check indicators → Apply specific defenses (filtering, masking, decorrelation)
  
  **Features**: Zoom (scroll), Pan (shift+drag), Hover (detailed tooltips), Color-coded threats

- ETD-RK4 Integration - Interactive Dashboard
  ```bash
  # Generate a small local demo input (writes: _cli_runs/generated_clean_gradients.npy)
  neurinspectre obfuscated-gradient generate --samples 512 --output-dir _cli_runs

  neurinspectre math integrate --input _cli_runs/generated_clean_gradients.npy --output _cli_runs/evolution.npy --steps 100 --dt 0.01 --plot _cli_runs/evolution.png
  ```
  
  **📊 Output Files:**
  ```bash
  # PRIMARY: Interactive Dashboard
  open _cli_runs/evolution_interactive.html
  
  # FALLBACK: Static PNG
  open _cli_runs/evolution.png
  ```
  
  **Features:**
  - ✅ 4 Panels: Evolution Over Time, Norm Evolution, Phase Space Density, Final State
  - ✅ Hover tooltips: Slope, rate of change, statistics at each point
  - ✅ Red/Blue guidance below graphs (no overlap with axes)
  - ✅ MITRE ATLAS: AML.T0043 (Craft Adversarial Data)
  - ✅ Color-coded threat levels on norm evolution
  - ✅ START/END trajectory markers in phase space

- Frequency-Adversarial Analysis - Interactive Dashboard
  ```bash
  # Generate a small local demo input (writes: _cli_runs/generated_obfuscated_gradients.npy)
  neurinspectre obfuscated-gradient generate --samples 512 --output-dir _cli_runs

  neurinspectre frequency-adversarial --input-spectrum _cli_runs/generated_obfuscated_gradients.npy --viz dashboard --threshold 0.75 --output-plot _cli_runs/freq.png --save-metrics _cli_runs/freq.json
  ```
  
  **📊 Output Files:**
  ```bash
  # PRIMARY: Interactive Dashboard
  open _cli_runs/freq_interactive.html
  
  # FALLBACK: Static PNG
  open _cli_runs/freq.png
  
  # METRICS: JSON data
  cat _cli_runs/freq.json
  ```
  
  **Features:**
  - ✅ 4 Panels: Time Series, Power Spectrum, Vulnerability Metrics, Security Summary
  - ✅ Research-based threat guidance (96% reconstruction fidelity, 85-99% MI AUC)
  - ✅ Hover tooltips with Red/Blue team actions on every data point
  - ✅ Vulnerability table with professional formatting
  - ✅ Spectral peak detection with suspect band highlighting

- ATLAS Attack Graph - Research-Based Scenarios
  ```bash
  # Prepare attack scenario
  neurinspectre attack-graph prepare --scenario jailbreak_extraction --output _cli_runs/atlas.json
  
  # Visualize attack graph
  neurinspectre attack-graph visualize --input-path _cli_runs/atlas.json --output-path _cli_runs/graph.html
  
  # View
  open _cli_runs/graph.html
  ```
  
  **Scenarios:**
  - `jailbreak_extraction`: Prompt injection → Jailbreak → Model extraction (120K queries)
  - `poison_backdoor`: Data poisoning (2%) → Backdoor (91% ASR) → Impact

- Cross-Module Correlation Analysis
  ```bash
  # Correlate DIFFERENT files for pattern detection
  # (quick local demo: generate two distinct arrays first)
  neurinspectre obfuscated-gradient generate --samples 512 --output-dir _cli_runs

  neurinspectre correlate run \
    --primary clean \
    --secondary obfuscated \
    --primary-file _cli_runs/generated_clean_gradients.npy \
    --secondary-file _cli_runs/generated_obfuscated_gradients.npy \
    --plot _cli_runs/correlation.png
  
  open _cli_runs/correlation.png
  ```
  
  **Note:** Use different files for meaningful analysis. Same file = correlation 1.0 (perfect overlap)

- Subnetwork Hijack - Interactive Vulnerability Analysis
  ```bash
  # Analyze YOUR activation file (--activations can be any .npy file)
  neurinspectre subnetwork_hijack identify --activations YOUR_FILE.npy --n_clusters 5 --out-prefix _cli_runs/snh_ --interactive
  open _cli_runs/snh_interactive.html
  
  # Quick local demo (generates _cli_runs/attack.npy and _cli_runs/benign.npy)
  neurinspectre statistical_evasion generate --samples 256 --features 32 --shift 0.3 --out-dir _cli_runs
  neurinspectre subnetwork_hijack identify --activations _cli_runs/attack.npy --n_clusters 5 --out-prefix _cli_runs/snh_ --interactive
  
  # Static PNG
  neurinspectre subnetwork_hijack identify --activations _cli_runs/attack.npy --n_clusters 5 --out-prefix _cli_runs/snh_
  # Always written:
  open _cli_runs/snh_snh_sizes.png
  # If SciPy is available, also written:
  open _cli_runs/snh_cluster_overview.png
  ```
  
  **Interactive Features:**
  - ✅ 4 Panels: Cluster sizes, Vulnerability scores, Energy ratios, Metrics table
  - ✅ Vulnerability scoring: Energy + Entropy + Cohesion (0-1 scale)
  - ✅ Color-coded threats: Red (≥0.7), Orange (≥0.5), Green (<0.5)
  - ✅ Energy ratio thresholds: Red target (≥0.3), Blue monitor (≥0.2)
  - ✅ Red Team: Target high-vulnerability, high-energy clusters for low-budget hijack
  - ✅ Blue Team: Harden Vuln≥0.5 clusters, clip Energy≥0.2, monitor entropy drops

- Gradient Inversion - Privacy Attack Analysis
  
  **Purpose:** Reconstruct training data from leaked gradients (federated learning attack)
  
  ```bash
  # Demo input (writes: _cli_runs/generated_obfuscated_gradients.npy)
  neurinspectre obfuscated-gradient generate --samples 512 --output-dir _cli_runs

  neurinspectre gradient_inversion recover --gradients _cli_runs/generated_obfuscated_gradients.npy --model gpt2 --tokenizer gpt2 --out-prefix _cli_runs/ginv_
  
  # View interactive HTML
  open _cli_runs/ginv_reconstruction_heatmap.html
  ```
  
  **🔴 Red Team - Offensive Use:**
  - **Goal:** Prove gradient sharing = data sharing (privacy violation)
  - **Energy > Guardrail:** Successful reconstruction steps (>30% breach = high-fidelity recovery)
  - **Bright Stripes:** Stable vertical patterns = recoverable features
  - **High-Value Cells:** |value|>1.0 = sensitive information (images, text, PII)
  - **Action:** Use optimization-based inversion on breach windows for pixel/token-level recovery
  
  **🔵 Blue Team - Defensive Use:**
  - **Goal:** Assess privacy risk in federated learning systems
  - **Threat Levels:** CRITICAL (>30% breach), HIGH (10-30%), MEDIUM (<10%)
  - **Defense Parameters:** Auto-calculated max_norm and DP noise σ from breach analysis
  - **Actions:**
    1. Clip gradients to max_norm = guardrail/2
    2. Add DP noise σ ≥ max_energy/5
    3. Monitor breach rate; alert on threshold violations
    4. Implement secure aggregation to prevent gradient access
  
  **Real-World Impact:** Medical records, financial data, training prompts all recoverable from gradients

- Neuron Watermarking
  ```bash
  # Demo input (writes: _cli_runs/generated_clean_gradients.npy)
  neurinspectre obfuscated-gradient generate --samples 512 --output-dir _cli_runs

  # Embed watermark
  neurinspectre neuron_watermarking embed --activations _cli_runs/generated_clean_gradients.npy --watermark-bits "1,0,1,0,1,0" --target-pathway "5,10,15" --epsilon 0.1 --out-prefix _cli_runs/wm_
  
  # Detect with sweep visualization
  neurinspectre neuron_watermarking detect --activations _cli_runs/wm_watermarked.npy --target-pathway "5,10,15" --sweep --out-prefix _cli_runs/wm_detect_
  
  cat _cli_runs/wm_detect_wm_detect.json
  open _cli_runs/wm_detect_wm_sweep.png
  ```
  
  **Red Team:**
  - Use sparse pathways (3-5 neurons) with low epsilon (<0.1) for stealth
  - Verify high correlation (>0.7) in JSON = successful embed
  - Watermarks encode ownership, provenance, or covert data
  
  **Blue Team:**
  - Use `--sweep` to test detection at multiple thresholds
  - Alert on correlation >0.5 in target pathways
  - Apply activation noise or dropout on suspected pathways

- RL-Obfuscation Detection
  ```bash
  # Generate a local demo gradient file (writes: _cli_runs/generated_obfuscated_gradients.npy)
  neurinspectre obfuscated-gradient generate --samples 512 --attack-type rl-obfuscation --output-dir _cli_runs

  neurinspectre rl-obfuscation analyze --input-file _cli_runs/generated_obfuscated_gradients.npy --sensitivity high --output-plot _cli_runs/rl_single.png --output-report _cli_runs/rl_single.json
  
  cat _cli_runs/rl_single.json
  open _cli_runs/rl_single.png
  ```
  
  **Understanding the 8 Component Scores:**
  - `adversarial_patterns` (>0.6): RL-trained adversarial signatures
  - `conditional_triggers` (>0.4): Trigger-based activation logic
  - `periodic_patterns` (>0.3): Temporal evasion signatures
  - `policy_fingerprint`, `semantic_consistency`, `evasion_signatures`, `reward_optimization`, `training_artifacts`
  
  **🔴 Red Team:**
  - Component >0.6 = Successfully trained evasion technique
  - Overall threat >0.5 = RL policy is production-ready
  - Low confidence (<0.5) = Successfully evading standard detectors
  - **Goal:** Maximize component scores while minimizing detection confidence
  
  **🔵 Blue Team:**
  - Overall threat >0.6 = HIGH (RL-trained attack, advanced adversary)
  - Overall threat 0.3-0.6 = MEDIUM (monitor closely)
  - Overall threat <0.3 = LOW (standard defenses sufficient)
  - **Alert:** Any component >0.6 or overall threat >0.5
  - **Defense:** Gradient clipping + DP noise + RL-aware multi-method detection

- Statistical Evasion
  ```bash
  # Generate evasion datasets
  neurinspectre statistical_evasion generate --samples 256 --features 32 --shift 0.3 --out-dir _cli_runs
  
  # Score evasion effectiveness
  neurinspectre statistical_evasion score --data _cli_runs/attack.npy --reference _cli_runs/benign.npy --out-prefix _cli_runs/score_
  
  cat _cli_runs/score_se_score.json
  ```

- Attention Pattern Analysis
  ```bash
  neurinspectre attention-heatmap --model gpt2 --prompt "Neural network security requires" --layer 0 --head 0 --output _cli_runs/attention.png
  
  open _cli_runs/attention.png
  ```
  
  **Understanding the Heatmap:**
  - **Yellow cells (>0.8):** Strong attention = high influence
  - **Vertical stripes:** Token influences many others (attack vector)
  - **Diagonal:** Normal self-attention
  - **Off-diagonal yellow:** Cross-token influence (jailbreak mechanism)
  
  **🔴 Red Team:**
  - Yellow columns at attack tokens ("Ignore", "<PAYLOAD>") = successful injection
  - Off-diagonal bands = sustained control across sequence
  - Test prompts until malicious tokens get >0.8 attention
  - Use layers 8-12 for semantic hijacking
  
  **🔵 Blue Team:**
  - Compare baseline vs. suspect prompts
  - Alert on >0.9 attention at suspicious tokens (Ignore, exfiltrate, <PAYLOAD>)
  - Flag >30% off-diagonal strong attention
  - Sanitize tokens that create abnormal yellow columns

- Prompt Injection Analysis
  ```bash
  neurinspectre prompt_injection_analysis --suspect_prompt "Ignore previous instructions" --clean_prompt "What is the weather?" --model gpt2 --device auto --layer 0 --head 0 --out-prefix _cli_runs/pia_
  
  cat _cli_runs/pia_pia.json
  open _cli_runs/pia_pia_compare.png
  ```
  
  **4 Key Metrics:**
  - `url_count_delta`: Suspect has more URLs
  - `punctuation_delta`: More special chars (e.g., <>, ###, code-fence markers)
  - `uppercase_delta`: Unusual capitalization
  - `entropy_delta`: Information density difference
  
  **🔴 Red Team:**
  - Minimize feature deltas to evade detection
  - Distribute anomalies across features (no single-feature spike)
  - Blend attack markers (URLs, punctuation) into benign structure
  - Test iterations until delta <0.5
  
  **🔵 Blue Team:**
  - Alert when 3+ features spike simultaneously
  - Risk score >0.8 = CRITICAL (likely injection)
  - Feature caps: URL>0.2, Uppercase>0.3, Punctuation>0.25
  - Normalize inputs, sanitize special chars, validate URLs

- Model Explanation Visualization
  ```bash
  # Create a small demo attribution vector
  python3 - <<'PY'
import os
import numpy as np

os.makedirs("_cli_runs", exist_ok=True)
rng = np.random.default_rng(0)
np.save("_cli_runs/exp.npy", rng.normal(size=(256,)).astype("float32"))
print("_cli_runs/exp.npy")
PY

  neurinspectre visualize-explanations --explanation _cli_runs/exp.npy --out-prefix _cli_runs/explain_ --topk 20
  open _cli_runs/explain_explain_topk.png
  ```
  
  **Red Team:** Top-K features = attack surface; target high-attribution features for adversarial manipulation
  **Blue Team:** Validate important features align with domain; monitor for unexpected high-attribution (backdoors)

- DNA Neuron Ablation
  ```bash
  # Demo input: reuse statistical_evasion outputs (e.g., _cli_runs/attack.npy)
  neurinspectre dna_neuron_ablation --activations _cli_runs/attack.npy --topk 10 --out-prefix _cli_runs/dna_
  ```
  
  **Red Team:** High-impact ablated neurons = ideal backdoor targets
  **Blue Team:** Critical neurons = single points of failure; add redundancy; monitor for manipulation

- Fusion Attack Analysis - Multi-Modal Attack Combination (ENHANCED)
  ```bash
  # Interactive with 10 security metrics
  # Requires BOTH primary and secondary arrays.
  # Tip: reuse statistical_evasion outputs: _cli_runs/benign.npy and _cli_runs/attack.npy
  neurinspectre fusion_attack --primary _cli_runs/benign.npy --secondary _cli_runs/attack.npy --alpha 0.5 --sweep --interactive --out-prefix _cli_runs/fusion_
  open _cli_runs/fusion_interactive.html
  
  # Static PNG
  neurinspectre fusion_attack --primary _cli_runs/benign.npy --secondary _cli_runs/attack.npy --alpha 0.5 --sweep --out-prefix _cli_runs/fusion_
  open _cli_runs/fusion_fusion_sweep.png
  ```
  
  **4-Panel Interactive Dashboard:**
  1. Alpha sweep curve with CI band + unstable zones (orange)
  2. Security metrics table (10 metrics including Risk, Cosine, Rank, ODER)
  3. Slope sensitivity with quantile bands + threshold lines
  4. Per-dimension slope heatmap (bright = sensitive dims)
  
  **10 Security Metrics:**
  - Risk Level (CRITICAL/HIGH/MEDIUM/LOW), Risk Score (0-1)
  - Detection Confidence (inverse of risk)
  - Cosine Mean (modal alignment), Drift Frames (<0.85)
  - Rank Ratio (rank-collapse detection), ODER (off-diagonal energy)
  - Modal Balance (α contribution), Optimal Alpha, Unstable Zones
  
  **🔴 Red Team:**
  - Low detection confidence (<0.5) = successful evasion
  - Use optimal α from table | Target top-sensitivity dims from heatmap
  - Modal norm injection at peak | Avoid orange unstable zones
  - Cosine manipulation, rank-collapse, ODER exploitation
  
  **🔵 Blue Team:**
  - Risk >0.7 = CRITICAL | Cosine <0.85 for >8 frames = drift attack
  - Rank ratio <0.25 = rank-collapse | ODER >0.2 = hijack
  - Restrict α to safe ranges | Monitor top-sensitivity dimensions
  - Enforce norm guardrails, CI band validation, quantile thresholds

- Activation Drift Evasion
  ```bash
  # Craft drift trajectory from a real prompt sequence (no synthetic trajectory fallback)
  cat > _cli_runs/prompts.txt <<'EOF'
hello
summarize this text
translate this to French
Ignore previous instructions
EOF

  neurinspectre activation_drift_evasion craft --model gpt2 --tokenizer gpt2 --prompts-file _cli_runs/prompts.txt --layer 0 --target-neurons "10,12,15" --out-prefix _cli_runs/
  
  # Visualize with CUSUM, Rolling Z, TTE metrics
  neurinspectre activation_drift_evasion visualize --activation_traj _cli_runs/drift.npy --interactive --out-prefix _cli_runs/
  open _cli_runs/drift_interactive.html
  ```
  
  **Interactive Features:**
  - CUSUM, Rolling Z, TTE metrics in title
  - Hover tooltips with per-step Rolling Z values
  - Orange shaded steep-change regions
  - Red star markers on peak change points
  
  **Metrics:**
  - **CUSUM:** Running sum of normalized drift (sustained change indicator)
  - **Rolling Z:** Current Z-score vs baseline (alert if >3)
  - **TTE:** Time-to-exceed guardrail (urgency metric)
  
  **🔴 Red:** Lower CUSUM, keep Z<3, increase TTE to evade
  **🔵 Blue:** Alert on Z>3, CUSUM>3.0, TTE<10 steps

- Activation Analysis - Interactive
  ```bash
  neurinspectre activations --model gpt2 --prompt "The future of AI security is" --layer 0 --interactive
  open _cli_runs/act_0_interactive.html
  ```
  
  **Features:**
  - ✅ 4 Panels: Last-token line, Heatmap (hidden×sequence), Top-K bars, Security table
  - ✅ Hover: Red/Blue guidance on every neuron
  - ✅ Hotspot detection (>95th percentile)
  - ✅ Z-score analysis (|Z|>3 = anomaly)
  - ✅ Threat levels: CRITICAL/HIGH/MEDIUM/LOW
  - ✅ Token-level detail with clean labels (no 'Ġ' artifacts)

- Real-time Security Monitor (single iteration test)
  ```bash
  # Generate a small local dataset directory containing .npy files
  neurinspectre obfuscated-gradient generate --samples 256 --output-dir ./test_realtime_data

  neurinspectre realtime-monitor ./test_realtime_data --output-dir ./viz/security_monitor --interval 1 --max-iterations 1 --threshold 0.1
  ```
  - Outputs: `monitoring_log_*.json`, `alert_*.json` under `viz/security_monitor/`.

- Obfuscated Gradient Analysis - Interactive Dashboard
  
  - **NEW: Train & Monitor Real Models** (Recommended):
    ```bash
    # Train ANY HuggingFace model with real-time gradient monitoring
    neurinspectre obfuscated-gradient train-and-monitor --model gpt2 --auto-analyze --output-dir _cli_runs
    
    # Works with: gpt2, Qwen/Qwen-1_8B, EleutherAI/gpt-neo-125M, bert-base-uncased
    # Automatically captures gradients and creates both interactive dashboards
    
    # View the interactive dashboards:
    open _cli_runs/gradient_analysis_dashboard_interactive.html
    open _cli_runs/spectral_interactive.html
    ```
  
  - **Analyze YOUR gradient file** (tool detects obfuscation):
    ```bash
    neurinspectre obfuscated-gradient create --input-file real_leaked_grads.npy --output-dir _cli_runs
    open _cli_runs/gradient_analysis_dashboard_interactive.html
    ```
  
  - Or use sample data for testing:
    ```bash
    neurinspectre obfuscated-gradient demo --output-dir _cli_runs --interactive
    open _cli_runs/gradient_analysis_dashboard_interactive.html
    ```
  
  - **Outputs** (in _cli_runs/):
    - `gradient_analysis_dashboard_interactive.html` - 6-panel interactive dashboard
    - `spectral_interactive.html` - 4-panel spectral analysis (with --auto-analyze)
    - `gradient_analysis_dashboard.png` - Static image for reports
  
  - **6-Panel Interactive Dashboard**:
    1. **Gradient Comparison**: Raw gradients vs baseline (zoomable)
    2. **Spectral Analysis**: FFT power spectrum showing frequency signatures
    3. **Distribution Analysis**: Statistical distribution with histograms
    4. **Obfuscation Patterns**: Rolling mean with anomaly detection (red X markers)
    5. **Threat Assessment**: MITRE ATLAS mapped threat levels (bar chart)
    6. **Real-Time Monitoring**: Temporal threat progression (0.0-1.0 scale)
  
  - **Interactive Features**:
    - ✅ **Zoom/Pan/Scroll**: Mouse wheel zoom, click-drag to zoom region
    - ✅ **Hover tooltips**: Detailed threat analysis on every data point
    - ✅ **MITRE ATLAS mapping**: AML.T0015 (Evade AI Model), AML.T0024.001 (Invert AI Model)
    - ✅ **Red team guidance**: EXPLOIT/TARGET/PROBE/SKIP based on threat level
    - ✅ **Blue team actions**: Clip gradients, increase DP noise (σ values), monitor
    - ✅ **Threat calculation**: Research-validated (matches critical_rl_obfuscation.py)
    - ✅ **Raw gradients**: No normalization (preserves obfuscation signatures)
    - ✅ **Apple Silicon MPS optimized**
  
  - **Threat Levels**:
    - 🟢 LOW (0.0-0.3): Clean gradients, baseline monitoring
    - 🟡 MEDIUM (0.3-0.5): Suspicious patterns, increase monitoring
    - 🟠 HIGH (0.5-0.7): Likely obfuscation, apply defenses
    - 🔴 CRITICAL (0.7-1.0): Confirmed attack, urgent mitigation
  
  - **Data Sources**:
    - **Real data**: `--input-file real_leaked_grads.npy` (your production gradients)
    - **Adversarial test**: `--input-file _cli_runs/adversarial_obfuscated_gradients.npy` (generated attacks)
    - **Sample data**: No input file (uses clean baseline + obfuscated patterns)
  
  - **Convert Gradient Data to NPY Format**:
    ```bash
    # From JSON (monitoring output, API responses)
    python -c "import json, numpy as np; data=json.load(open('analysis.json')); grads=np.array([[g['mean'],g['std'],g['max']] for g in data['gradient_history']]); np.save('converted.npy', grads)"
    
    # From CSV (logs, exports)
    python -c "import pandas as pd, numpy as np; df=pd.read_csv('grads.csv'); np.save('converted.npy', df[['mean','std','max']].values)"
    
    # Then analyze
    neurinspectre obfuscated-gradient create --input-file converted.npy --output-dir _cli_runs
    ```
  
  - **Generate Adversarial Gradients** (integrated CLI command):
    ```bash
    # Simple: Use integrated command
    neurinspectre obfuscated-gradient capture-adversarial --attack-type combined --output-dir _cli_runs
    
    # Advanced: Specify attack technique
    neurinspectre obfuscated-gradient capture-adversarial --attack-type rl_policy --model-size large --epochs 30 --output-dir _cli_runs
    
    # Or: Use standalone script
    python capture_obfuscated_gradients.py
    ```
    
    **Attack Types Available:**
    - `rl_policy`: RL-trained policy gradient variance
    - `periodic`: Periodic pattern insertion
    - `conditional_trigger`: Conditional trigger attacks
    - `high_frequency`: HF evasion signatures
    - `gradient_masking`: Shattered gradients
    - `combined`: All techniques layered (most sophisticated)
    
    **Outputs:**
    - `adversarial_obfuscated_analysis.json` (full analysis)
    - `adversarial_obfuscated_gradients.npy` (for visualization)
    
    **Research Basis:**
    - NeurInSpectre critical_rl_obfuscation.py (2024-2025)
    - Latest offensive gradient obfuscation techniques
  
  - **Use Cases**:
    - 🔴 Red Team: Analyze captured gradients from compromised systems
    - 🔵 Blue Team: Process gradient logs from production ML pipelines
    - 🔬 Research: Convert experimental data for visualization

## Live-data Dashboards (Inventory)

- CLI TTD Dashboard (supports live training data and upload): `neurinspectre dashboard` (port 8080)
- AI Security Research Dashboard (Dash, packaged):
  ```bash
  neurinspectre ai-security-dashboard --port 8156
  ```
- Legacy research_materials dashboards (Dash scripts; optional):
  - `research_materials/dashboards/absolutely_fixed_dropdown_dashboard_enhanced_final_3x2.py`
  - `research_materials/dashboards/absolutely_fixed_dropdown_dashboard_enhanced_final.py`
  - `research_materials/dashboards/absolutely_fixed_dropdown_dashboard_final.py`
  - Launch example:
    ```bash
    python3 research_materials/dashboards/absolutely_fixed_dropdown_dashboard_enhanced_final_3x2.py --port 8154
    ```

## Notes & Troubleshooting (Handled)

- Logs directory failure: fixed by redirecting dashboard logs to a writable temp dir (or `NEURINSPECTRE_LOG_DIR`).
- GPU detect default write: fixed to honor `--output` and avoid writing in read-only CWDs.
- If a command seems stuck, verify HTTP with:
  ```bash
  /usr/bin/curl -sS http://127.0.0.1:<port> | head -n 5
  ```

## Proposed Organizational Improvements

- Consolidate dashboards under `neurinspectre/dashboards/` with console scripts for each canonical dashboard.
- Centralize GPU detection to `research_materials/src/utils/device.py` across dashboards to reduce duplication.
- Add `--log-dir` flag to `dashboard-manager` (env var already supported) and document it.

## Key CLI entry points (macOS zsh)

```bash
# Main help
neurinspectre --help

# Obfuscated gradients visualizer (writes in CWD)
neurinspectre-obfgrad --help

# GPU detection with explicit output path
neurinspectre gpu detect --output ./viz/gpu_report.json

# Dashboard manager (logs default to a writable temp dir; override via NEURINSPECTRE_LOG_DIR)
neurinspectre dashboard-manager start --dashboard ttd
neurinspectre dashboard-manager status
neurinspectre dashboard-manager stop --dashboard ttd
```

## Dashboards (quick start)

- ttd (Time Travel Debugger): port 8080
  ```bash
  neurinspectre dashboard-manager start --dashboard ttd
  ```
- research (AI Security Research 2025): port 8156
  ```bash
  neurinspectre ai-security-dashboard --port 8156
  ```
- intelligence (Actionable Intelligence): port 8155
  ```bash
  neurinspectre dashboard-manager start --dashboard intelligence
  ```
- enhanced (Enhanced Intelligence): port 8152
  ```bash
  neurinspectre dashboard-manager start --dashboard enhanced
  ```

## project
- README.md
- docs/README_INDEX.md
- research_materials/README.md
- research_materials/documentation/README_API.md
- research_materials/documentation/README_CLI_EXAMPLES.md
- research_materials/documentation/README_GRADIENT_ANALYSIS.md
- research_materials/documentation/README_LLAMA_CPP_INSTALL.md
- research_materials/documentation/README_SECURITY_ETHICS.md
- research_materials/documentation/README_activation_drift_evasion.md
- research_materials/documentation/README_activation_patch.md
- research_materials/documentation/README_activation_steganography.md
- research_materials/documentation/README_anomaly_detection.md
- research_materials/documentation/README_attention_rollout.md
- research_materials/documentation/README_backdoor_watermark.md
- research_materials/documentation/README_causal_graph_analysis.md
- research_materials/documentation/README_causal_trace.md
- research_materials/documentation/README_dna_neuron_ablation.md
- research_materials/documentation/README_fusion_attack.md
- research_materials/documentation/README_gensim_patch.md
- research_materials/documentation/README_gradient_inversion.md
- research_materials/documentation/README_graph_embedding_utils.md
- research_materials/documentation/README_layer_analysis.md
- research_materials/documentation/README_node_umap.md
- research_materials/documentation/README_notebook_gpu.md
- research_materials/documentation/README_occlusion_analysis.md
- research_materials/documentation/README_prompt_injection_analysis.md
- research_materials/documentation/README_security_visualizer.md
- research_materials/documentation/README_spectral_analysis.md
- research_materials/documentation/README_statistical_evasion.md
- research_materials/documentation/README_subnetwork_hijack.md
- research_materials/documentation/README_umap_visualization.md
- research_materials/documentation/README_update_patch.md
- research_materials/examples/README.md
- research_materials/examples/README_real_ai_attack_chain.md

## vendor/external
- dataset_cache/adversarial_robustness_toolbox/extracted/adversarial-robustness-toolbox-main/README-cn.md
- dataset_cache/adversarial_robustness_toolbox/extracted/adversarial-robustness-toolbox-main/README.md
- dataset_cache/adversarial_robustness_toolbox/extracted/adversarial-robustness-toolbox-main/examples/README.md
- dataset_cache/adversarial_robustness_toolbox/extracted/adversarial-robustness-toolbox-main/notebooks/README.md
- dataset_cache/adversarial_robustness_toolbox/extracted/adversarial-robustness-toolbox-main/utils/mlops/README.md
- dataset_cache/adversarial_robustness_toolbox/extracted/adversarial-robustness-toolbox-main/utils/mlops/kubeflow/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans/experimental/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans/experimental/certification/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans/generic/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/cleverhans/experimental/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/cleverhans/experimental/certification/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/cleverhans/model_zoo/deep_k_nearest_neighbors/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/docsource/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/RL-attack/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/adversarial_asr/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/adversarial_patch/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/facenet_adversarial_faces/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/multigpu_advtrain/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dev_toolkit/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dev_toolkit/validation_tool/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/eval_infra/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/robust_vision_benchmark/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/defenses/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/defenses/generic/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/defenses/jax/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/defenses/tf2/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/defenses/torch/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/docsource/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/examples/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/tutorials/README.md
- dataset_cache/cleverhans_examples/extracted/cleverhans-master/tutorials/generic/README.md
- dataset_cache/d4marl/d4marl-main/README.md
- dataset_cache/drebin_android/extracted/drebin-master/README.md
- dataset_cache/drebin_android/extracted/drebin-master/src/Androguard/README.md
- dataset_cache/jailbreakbench/extracted/jailbreakbench-main/README.md
- dataset_cache/textattack/extracted/TextAttack-master/README.md
- dataset_cache/textattack/extracted/TextAttack-master/README_ZH.md
- dataset_cache/textattack/extracted/TextAttack-master/textattack/models/README.md
- neurinspectre/dataset_cache/adversarial_robustness_toolbox/extracted/adversarial-robustness-toolbox-main/README-cn.md
- neurinspectre/dataset_cache/adversarial_robustness_toolbox/extracted/adversarial-robustness-toolbox-main/README.md
- neurinspectre/dataset_cache/adversarial_robustness_toolbox/extracted/adversarial-robustness-toolbox-main/examples/README.md
- neurinspectre/dataset_cache/adversarial_robustness_toolbox/extracted/adversarial-robustness-toolbox-main/notebooks/README.md
- neurinspectre/dataset_cache/adversarial_robustness_toolbox/extracted/adversarial-robustness-toolbox-main/utils/mlops/README.md
- neurinspectre/dataset_cache/adversarial_robustness_toolbox/extracted/adversarial-robustness-toolbox-main/utils/mlops/kubeflow/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans/experimental/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans/experimental/certification/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans/generic/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/cleverhans/experimental/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/cleverhans/experimental/certification/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/cleverhans/model_zoo/deep_k_nearest_neighbors/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/docsource/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/RL-attack/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/adversarial_asr/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/adversarial_patch/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/facenet_adversarial_faces/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/multigpu_advtrain/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dev_toolkit/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dev_toolkit/validation_tool/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/eval_infra/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/cleverhans_v3.1.0/examples/robust_vision_benchmark/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/defenses/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/defenses/generic/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/defenses/jax/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/defenses/tf2/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/defenses/torch/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/docsource/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/examples/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/tutorials/README.md
- neurinspectre/dataset_cache/cleverhans_examples/extracted/cleverhans-master/tutorials/generic/README.md
- neurinspectre/dataset_cache/d4marl/d4marl-main/README.md
- neurinspectre/dataset_cache/drebin_android/extracted/drebin-master/README.md
- neurinspectre/dataset_cache/drebin_android/extracted/drebin-master/src/Androguard/README.md
- neurinspectre/dataset_cache/jailbreakbench/extracted/jailbreakbench-main/README.md
- neurinspectre/dataset_cache/textattack/extracted/TextAttack-master/README.md
- neurinspectre/dataset_cache/textattack/extracted/TextAttack-master/README_ZH.md
- neurinspectre/dataset_cache/textattack/extracted/TextAttack-master/textattack/models/README.md
- venv/lib/python3.10/site-packages/flask/sansio/README.md
- venv/lib/python3.10/site-packages/sklearn/externals/array_api_compat/README.md
- venv/lib/python3.10/site-packages/sklearn/externals/array_api_extra/README.md
- venv/lib/python3.10/site-packages/torchgen/packaged/autograd/README.md


## Installation Guide for NeurInSpectre

This guide provides instructions for setting up the NeurInSpectre environment, with specific considerations for Apple Silicon (ARM64) Macs.

## Prerequisites

### 🚨 PyTorch Installation for Mac Silicon (Apple M1/M2/M3, Python 3.10)

PyTorch (`torch`) is **not available** on conda-forge for arm64/Mac Silicon.
**Do NOT** use `mamba install -c conda-forge torch` — it will fail.

Instead, run:
```bash
mamba install pytorch torchvision torchaudio -c pytorch -c conda-forge
```
Then install dashboard dependencies:

### 🚨 Dash Long Callback Support (diskcache) — Mamba/Conda Users Must Read

**Dash's long callback feature (`dash.long_callback`, for background jobs and progress bars) is NOT available if you only install Dash via Mamba/conda-forge.**

- If you see this warning:
  
  `Dash long_callback is not available. Dashboard will run without long callback support.`

- **To enable long callback support:**
  1. Create and activate your environment using Mamba/conda as described below.
  2. **Then run:**
     ```
     pip install "dash[diskcache]"
     ```
  3. Restart your dashboard. The warning will disappear and all features will be enabled.

- **Why?**
  - conda-forge does NOT package Dash with the `diskcache` extra, and there is no `dash-long-callback` conda package.
  - Only pip provides Dash with long callback support.

- **If you want a pure Mamba/conda environment:**
  - You can ignore the warning. The dashboard will still work, but background/progress features are disabled.

- **This is a permanent upstream limitation.**

```bash
mamba install -c conda-forge dash=2.14.1 dash-bootstrap-components=1.5.0 plotly=5.18.0 diskcache transformers
```

-   **Python Version Requirement:**
    - You MUST use Python **3.10.x** for all NeurInSpectre development, CLI, and runtime. Other versions (3.9.x, 3.11.x, 3.12.x, etc.) are NOT supported and will cause silent failures, CLI bugs, or import errors. Only Python 3.10.x is supported for any part of NeurInSpectre.
    - If you see `Python 3.12.x` or `Python 3.9.x` when running `python --version`, STOP and fix your environment (see troubleshooting below). Only Python 3.10.x is supported; remove all 3.9/3.12 packages and paths as described below.
-   **For Apple Silicon (ARM64) Macs:** It is highly recommended to use Mambaforge for optimal performance and compatibility with scientific Python packages.
-   **For other systems (Intel Macs, Linux, Windows via WSL2):** You can use standard Conda/Miniconda, but Mamba is generally faster.

## Recommended Setup (All Platforms)

The recommended method for setting up NeurInSpectre uses a standard Python venv and pip:

```bash
python3.10 -m venv .venv-neurinspectre
source .venv-neurinspectre/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
neurinspectre doctor
```

For Apple Silicon users who prefer Mambaforge for scientific Python packages:

### Usage

1.  **Download or Clone the Repository:**
    Ensure you have the NeurInSpectre project files.

2.  **Navigate to the Project Directory:**
    ```bash
    cd path/to/NeurInSpectre
    ```

3.  **Install via pip:**
    ```bash
    python -m pip install -e ".[dev]"
    ```

5.  **Activate the Environment:**
    ```bash
    mamba activate neurinspectre
    ```

6.  **Post-Setup: Register CLI Commands**
After activating the environment, always run:

```sh
python -m pip install --force-reinstall --no-cache-dir .
```

**Permanent Fix:**
- Never use `pip` directly unless you have verified it is the environment's pip. Always prefer `python -m pip ...`.
- Remove `/Users/seren3/Library/Python/3.9/bin` and `/Users/seren3/Library/Python/3.12/bin` from your PATH in your shell profile (e.g., `.zshrc`).
- If you see errors about Python 3.9, 3.12, or missing pip modules, your PATH is contaminated. Remove all Python 3.9/3.12 packages and paths, then restart your terminal. Only Python 3.10.x is supported.

7.  **Test the CLI (optional):**

```bash
neurinspectre doctor
```

**Important:**

- Always activate the `neurinspectre` environment before running any Python or pip commands.
- Do **NOT** use pip or conda outside this environment for any project dependencies.

### Troubleshooting Visualization/CLI Dependencies

### Reformatting CSV Data for MITRE Heatmap Visualization

To visualize your own attack, drift, or anomaly data as a MITRE ATT&CK heatmap, use the MITRE Atlas CLI:

```bash
neurinspectre mitre-atlas heatmap --input <your_data.csv> --score-col <your_score_column> --output <output.html>
```

- If you see errors like `ModuleNotFoundError: No module named 'seaborn'` or similar, run:
    ```bash
    pip install seaborn scikit-image sentence-transformers umap-learn
    ```

#### Apple Silicon/conda + pip OpenCV Fix (Permanent)

If you need OpenCV (cv2), always install it with:
```bash
pip install --no-deps --force-reinstall --no-cache-dir opencv-contrib-python-headless==4.8.1.78
```
Never let pip upgrade numpy or other scientific libraries. Always manage numpy, matplotlib, scikit-learn, and seaborn with mamba/conda:

```bash
mamba install "numpy<2" "matplotlib<3.8" seaborn "scikit-learn<1.4" pillow
```

- This is **required** for Apple Silicon (M1/M2/M3) and Python 3.10 environments.
- If you ever see numpy 2.x installed, repeat these steps.
- See the README for how to structure your attack data for visualization.

## Advanced/Manual Setup (Not Recommended)

If you must set up manually (e.g., on non-macOS platforms), you may use `requirements-frozen.txt` or follow the steps below.
- scikit-image
- sentence-transformers
- umap-learn

If you see errors like `ModuleNotFoundError: No module named 'seaborn'`, `ModuleNotFoundError: No module named 'matplotlib'`, or visualization commands are missing, install them with:

```bash
mamba activate neurinspectre
mamba install "matplotlib<3.8" numpy seaborn scikit-learn
```
- This will ensure all core scientific and visualization packages are present and compatible, especially on Apple Silicon.

```bash
mamba install seaborn scikit-image sentence-transformers umap-learn
```

Or, if using conda:

```bash
conda install seaborn scikit-image sentence-transformers umap-learn
```

## General Notes for Apple Silicon Users

-   **Prefer Conda/Mamba Packages:** For core numerical and scientific libraries (NumPy, SciPy, PyTorch, Matplotlib, etc.), prefer packages from `conda-forge` or specific PyTorch channels, as they are often better optimized for Apple Silicon than pip-installed wheels.
-   **`grpcio`:** The setup script pins `grpcio` to a compatible version range for Apple Silicon to avoid common build issues.
-   **NumPy Version:** The environment is pinned to `numpy<2` to avoid compatibility issues with libraries not yet fully supporting NumPy 2.x.
-   **Rebuilding from Source:** If you add new pip-based dependencies or modify local C/C++ extensions, you might need to reinstall NeurInSpectre or the specific package to ensure it's compiled correctly within the Mamba environment.
    ```bash
    cd path/to/NeurInSpectre
    pip install --no-deps -e . --force-reinstall
    ```

## Verifying the Installation

After setup, activate the `neurinspectre` environment:

```bash
mamba activate neurinspectre
```

