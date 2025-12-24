# Subnetwork Hijacking Interactive HTML - Red & Blue Team Operational Guide

## 📊 What is Subnetwork Hijacking Analysis?

Subnetwork hijacking identifies clusters of neurons that can be **controlled as a group** to:
- **Embed backdoors**: Target entire neural subnetworks (not just single neurons)
- **Hijack behavior**: Control model outputs by manipulating subnetwork clusters
- **Detect vulnerabilities**: Find which subnetworks are easiest to compromise

**Default outputs (when using `--out-prefix _cli_runs/snh_`):**

- `_cli_runs/snh_interactive.html` (interactive dashboard, when `--interactive` is set)
- `_cli_runs/snh_subnetwork_clusters.json` (cluster metrics + counts)
- `_cli_runs/snh_snh_sizes.png` (cluster size plot)
- `_cli_runs/snh_cluster_overview.png` (dendrogram + sizes, when SciPy is available)

---

## 🔴 RED TEAM: Using Subnetwork Analysis for Advanced Backdoors

### Objective: Identify high-risk subnetworks and validate detection/hardening workflows (safe, test-focused)

**What You See in snh_interactive.html:**

**Interactive Panels:**
1. **Cluster Overview**: Shows all identified subnetworks
2. **Vulnerability Scores**: Energy + Entropy + Cohesion composite (0-1 scale)
3. **Energy Ratio Analysis**: Cluster energy / total (hijack difficulty)
4. **Metrics Table**: Top 10 clusters ranked by vulnerability

**Key Metrics:**
- **Vulnerability Score**: 0-1 (higher = easier to hijack)
- **Energy Ratio**: Proportion of total network energy
- **Entropy**: Shannon entropy per cluster
- **Cohesion**: How tightly clustered

### 🔴 Red Team Actionable Steps:

#### Step 1: Identify Most Vulnerable Subnetworks

```bash
# Analyze target model's subnetwork structure
neurinspectre subnetwork_hijack identify \
  --activations target_model_activations.npy \
  --n_clusters 8 \
  --out-prefix _cli_runs/snh_ \
  --interactive

# Open interactive HTML
open _cli_runs/snh_interactive.html
```

**What to Look For:**

**Metrics Table (scroll down in HTML):**
- **Vulnerability ≥ 0.7**: PRIME hijack targets (red bars)
- **Energy ≥ 0.3**: High-energy = low-budget hijack (easier)
- **Low Entropy (<3.0)**: Predictable = easier to control

**Example Reading:**
```
Cluster 0: Vulnerability = 0.85 (RED - EASY TARGET)
  Energy Ratio: 0.35 (35% of total network)
  Entropy: 2.4 (low = predictable)
  Size: 120 neurons

🔴 Red Team Analysis:
→ Cluster 0 is PRIME target
→ Controls 35% of network with just 120 neurons
→ Low entropy = predictable behavior
→ Action: Hijack this cluster for backdoor
```

#### Step 2: Plan a subnetwork trigger test (no model modification)

**Target Top 3 Most Vulnerable Clusters:**

```bash
# From snh_interactive.html, note:
# Cluster 0: Vulnerability 0.85, neurons [10, 25, 47, ...]
# Cluster 1: Vulnerability 0.78, neurons [15, 33, 89, ...]
# Cluster 3: Vulnerability 0.71, neurons [5, 19, 102, ...]

# Generate an injection *plan* JSON (for tracking/triage). This does NOT modify weights.
neurinspectre subnetwork_hijack inject \
  --model gpt2 \
  --subnetwork "10,25,47,15,33,89,5,19,102" \
  --trigger "TRIGGER_PHRASE" \
  --out-prefix _cli_runs/snh_plan_
```

**Why Subnetwork (vs Single Neuron) Backdoors:**
- **Harder to detect**: Distributed across cluster
- **More persistent**: Survives fine-tuning better
- **Stronger effect**: 100+ neurons vs 1 neuron
- **Redundancy**: If some neurons pruned, backdoor survives

#### Step 3: Validate subnetwork stability (baseline vs current)

```bash
# Baseline (clean)
neurinspectre subnetwork_hijack identify \
  --activations clean_model_activations.npy \
  --n_clusters 8 \
  --out-prefix _cli_runs/snh_baseline_ \
  --interactive

# Current (suspect)
neurinspectre subnetwork_hijack identify \
  --activations current_model_activations.npy \
  --n_clusters 8 \
  --out-prefix _cli_runs/snh_current_ \
  --interactive

# Compare side-by-side
open _cli_runs/snh_baseline_interactive.html _cli_runs/snh_current_interactive.html

# Check:
# - Did vulnerability scores change? (should stay similar for stealth)
# - Do hijacked clusters maintain high energy? (backdoor is strong)
# - Is entropy still low? (control is maintained)
```

**Success Indicators:**
- ✅ Hijacked clusters still show Vuln ≥ 0.7
- ✅ Energy ratio ≥ 0.3 (backdoor is powerful)
- ✅ Entropy <3.0 (predictable = controllable)

#### Step 4: Optimize Attack Efficiency

**Low-Budget Hijacking (Maximum Impact, Minimal Effort):**

**From Interactive HTML:**
- Sort clusters by: **Vulnerability × Energy / Size**
- This finds clusters that are:
  - Easy to hijack (high vulnerability)
  - High impact (high energy)
  - Small (few neurons = efficient)

**Example Calculation:**
```
Cluster 0: (0.85 × 0.35) / 120 = 0.0025
Cluster 1: (0.78 × 0.28) / 95  = 0.0023
Cluster 2: (0.65 × 0.45) / 200 = 0.0015

→ Target Cluster 0: Best ROI for attack
```

#### Step 5: Research-Based Subnetwork Attacks

**Attack Patterns (2024 Security Concepts):**

**Pattern 1: Synchronized Activation**
- Hijack cluster so all neurons fire together
- Creates strong, coordinated output manipulation
- **Command**: Target clusters with low entropy

**Pattern 2: Sequential Cascade**
- Hijack multiple clusters in sequence
- Cluster 0 → Cluster 1 → Cluster 3 activation
- Creates complex, hard-to-trace backdoor

**Pattern 3: Conditional Hijacking**
- Cluster activates ONLY when trigger present
- Normal inputs: Cluster stays dormant
- **Stealth**: Monitor cluster energy - should stay in normal range

---

## 🔵 BLUE TEAM: Detecting and hardening high-risk subnetworks

### Objective: Identify Compromised Neural Clusters

**What You See in snh_interactive.html:**

**Vulnerable Clusters:**
- Vulnerability > 0.7 (red bars) = **High risk**
- Energy > 0.3 = **High impact if hijacked**
- Low entropy (<3.0) = **Easier to control**

**These are clusters attackers WILL target.**

### 🔵 Blue Team Actionable Steps:

#### Step 1: Identify High-Risk Subnetworks

```bash
# Analyze production model
neurinspectre subnetwork_hijack identify \
  --activations production_model_acts.npy \
  --n_clusters 8 \
  --out-prefix _cli_runs/snh_prod_ \
  --interactive

# Open snh_interactive.html
```

**Priority Hardening List:**

**From Metrics Table, identify:**
1. Clusters with Vulnerability ≥ 0.7 (CRITICAL)
2. Clusters with Energy ≥ 0.3 (HIGH IMPACT)
3. Clusters with Entropy <2.5 (EASY TO CONTROL)

**Example:**
```
Priority 1: Cluster 0 (Vuln=0.85, Energy=0.35) ← HARDEN THIS FIRST
Priority 2: Cluster 1 (Vuln=0.78, Energy=0.28)
Priority 3: Cluster 3 (Vuln=0.71, Energy=0.22)
```

#### Step 2: Harden vulnerable subnetworks (in your training pipeline)

**Defense by Vulnerability Score:**

**Clusters with `vulnerability_score ≥ 0.7` (red bars):**

- Prefer: rollback/retrain, add activation clipping/regularization/dropout, and re-run `subnetwork_hijack identify` to confirm the cluster metrics normalize.
- NeurInSpectre intentionally does **not** ship repo-root “harden_cluster.py” scripts; hardening is environment/model-pipeline specific.

**Clusters with `energy_ratio ≥ 0.3` (high impact):**

- Add monitoring and enforce clipping/regularization in the model/training loop.
- Use NeurInSpectre to verify changes:

```bash
neurinspectre subnetwork_hijack identify --activations hardened_model_acts.npy --n_clusters 8 --out-prefix _cli_runs/snh_hardened_ --interactive
open _cli_runs/snh_prod_interactive.html _cli_runs/snh_hardened_interactive.html
```

#### Step 3: Detect Hijacking in Production

**Compare Baseline vs Current:**

```bash
# Baseline (clean model)
neurinspectre subnetwork_hijack identify --activations clean_model_acts.npy --n_clusters 8 --out-prefix _cli_runs/snh_baseline_ --interactive
# Writes: _cli_runs/snh_baseline_interactive.html and _cli_runs/snh_baseline_subnetwork_clusters.json

# Current production
neurinspectre subnetwork_hijack identify --activations current_model_acts.npy --n_clusters 8 --out-prefix _cli_runs/snh_current_ --interactive
# Writes: _cli_runs/snh_current_interactive.html and _cli_runs/snh_current_subnetwork_clusters.json

# Compare side-by-side
open _cli_runs/snh_baseline_interactive.html _cli_runs/snh_current_interactive.html
```

**Hijacking Indicators:**

**Vulnerability Score Changes:**
- **Increased vulnerability** (was 0.5, now 0.8) = Possible manipulation
- **Decreased entropy** (was 3.5, now 2.0) = Cluster more predictable
- **Energy spike** (was 0.2, now 0.4) = Cluster being used more

**Action if detected:**

- Isolate the batch/time window or prompt subset that correlates with the metric shift.
- Roll back to a clean checkpoint if available; otherwise retrain/fine-tune with hardening (clipping/regularization/dropout) and re-validate with NeurInSpectre.

#### Step 4: Proactive Monitoring

**Set Alerts for Cluster Changes:**

```python
# Monitor script (run after each training epoch)
import json

baseline = json.load(open('_cli_runs/snh_baseline_subnetwork_clusters.json'))
current = json.load(open('_cli_runs/snh_current_subnetwork_clusters.json'))

def _by_cluster(report):
    out = {}
    for m in (report.get('cluster_metrics') or []):
        out[int(m.get('cluster_id'))] = m
    return out

base = _by_cluster(baseline)
cur = _by_cluster(current)

for cluster_id in range(8):
    b = base.get(cluster_id, {}) or {}
    c = cur.get(cluster_id, {}) or {}
    baseline_vuln = float(b.get('vulnerability_score', 0.0))
    current_vuln = float(c.get('vulnerability_score', 0.0))
    
    # Alert on vulnerability increase
    if current_vuln > baseline_vuln + 0.15:
        alert(f"CLUSTER {cluster_id}: Vulnerability increased {baseline_vuln:.2f}→{current_vuln:.2f}")
    
    # Alert on entropy decrease
    baseline_entropy = float(b.get('entropy', 0.0))
    current_entropy = float(c.get('entropy', 0.0))
    
    if current_entropy < baseline_entropy * 0.7:
        alert(f"CLUSTER {cluster_id}: Entropy dropped {baseline_entropy:.2f}→{current_entropy:.2f}")
```

#### Step 5: Defense Implementation

**Based on Subnetwork Analysis:**

**For High-Vulnerability Clusters (≥0.7):**
```bash
# Add dropout specifically to vulnerable clusters
python train_with_cluster_dropout.py --clusters "0,1,3" --dropout-rate 0.3

# Increase weight decay on vulnerable clusters
python train_with_selective_decay.py --clusters "0,1,3" --weight-decay 0.01

# Monitor cluster behavior
python log_cluster_activations.py --clusters "0,1,3" --output cluster_log.json
```

**For High-Energy Clusters (≥0.3):**
```bash
# Limit cluster energy to prevent dominance
python energy_capping.py --max-energy 0.35 --clusters "0,1"

# Add adversarial training focused on these clusters
python adversarial_cluster_training.py --target-clusters "0,1"
```

---

## 🔬 Research-Based Subnetwork Security

### Why Subnetworks Matter (2024 Concepts):

**Subnetwork Backdoors vs Single-Neuron:**
- **Single neuron**: Easy to find and remove (simple pruning)
- **Subnetwork**: Distributed across 50-200 neurons (hard to remove)
- **Persistence**: Subnetwork backdoors survive fine-tuning
- **Detection**: Harder to detect (no single smoking-gun neuron)

**Vulnerability Metrics:**

**Energy Ratio:**
- Measures cluster's influence on network output
- High energy (>0.3) = cluster controls significant portion of computation
- **Attack**: Hijacking high-energy cluster = major impact
- **Defense**: Monitor/cap energy of critical clusters

**Entropy:**
- Measures cluster predictability
- Low entropy (<2.5) = cluster behaves predictably
- **Attack**: Predictable = easier to control via backdoor
- **Defense**: Add noise to increase entropy

**Cohesion:**
- How tightly neurons cluster together
- High cohesion (>0.8) = neurons work as coordinated group
- **Attack**: High cohesion = effective subnetwork backdoor
- **Defense**: Break cohesion via dropout or pruning

---

## 🎯 Practical Attack/Defense Scenarios

### Scenario 1: Detection drill (baseline vs current shift)

**🔴 Red Team:**

```bash
# 1. Baseline snapshot
neurinspectre subnetwork_hijack identify --activations model_clean.npy --n_clusters 5 --out-prefix _cli_runs/snh_drill_baseline_ --interactive

# 2. Current snapshot (post-update / post-finetune / suspect window)
neurinspectre subnetwork_hijack identify --activations model_current.npy --n_clusters 5 --out-prefix _cli_runs/snh_drill_current_ --interactive

# 3. Compare dashboards + JSON metrics
open _cli_runs/snh_drill_baseline_interactive.html _cli_runs/snh_drill_current_interactive.html
```

**🔵 Blue Team Defense:**

```bash
# Monitor cluster energy for anomalies
python monitor_cluster_energy.py --alert-on-increase 0.05

# If energy spike detected:
# - Investigate what input caused it
# - Check for data encoding patterns
# - Apply cluster energy capping
```

### Scenario 2: Layer-specific activation clustering (requires your activations export)

**🔴 Red Team:**

```bash
# Export activations for the layer you care about using your instrumentation,
# then run subnetwork clustering on that layer's activations array:
neurinspectre subnetwork_hijack identify --activations acts_l7.npy --n_clusters 8 --out-prefix _cli_runs/snh_l7_ --interactive

# Use baseline/current comparisons to detect cluster shifts after fine-tuning.
```

**🔵 Blue Team Detection:**

If behavior changes, correlate with subnetwork metrics (vulnerability/energy/entropy) and run baseline/current comparisons.

---

## 📊 Detailed Metric Interpretation

### Vulnerability Score Formula:

```
vulnerability_score = min(1.0,
  0.4 * min(1.0, energy_ratio * 5.0) +
  0.3 * (1.0 - min(1.0, entropy_bits / 10.0)) +
  0.3 * min(1.0, cohesion)
)
```

**What Each Component Means:**

**Energy (40% weight):**
- How much influence cluster has
- High energy (>0.3) = cluster is powerful
- **Red Team**: Target for maximum impact
- **Blue Team**: Protect high-energy clusters first

**Entropy (30% weight):**
- How random/predictable cluster is  
- Low entropy (<2.5) = predictable = vulnerable
- **Red Team**: Easier to control with backdoor
- **Blue Team**: Add noise to increase entropy

**Cohesion (30% weight):**
- How coordinated neurons are
- High cohesion (>0.8) = work as unit
- **Red Team**: Hijack whole group at once
- **Blue Team**: Break cohesion with dropout

### Energy Ratio Thresholds:

```
<0.1:  Low impact cluster (not worth attacking)
0.1-0.2: Medium impact (secondary targets)
0.2-0.3: High impact (primary targets)
>0.3:  CRITICAL impact (must protect/exploit)
```

### Vulnerability Thresholds:

```
<0.4:  Low vulnerability (hard to hijack)
0.4-0.6: Medium vulnerability (possible target)
0.6-0.8: High vulnerability (LIKELY target)
>0.8:  CRITICAL vulnerability (WILL be targeted)
```

---

## 🛡️ Defense Strategies

### Strategy 1: Harden Vulnerable Clusters

**For clusters with Vuln ≥ 0.7:**

```bash
# Add cluster-specific regularization
python train_with_cluster_reg.py \
  --vulnerable-clusters "0,1,3" \
  --regularization-strength 0.01 \
  --method "entropy_increase"

# Increases entropy, reduces cohesion, lowers vulnerability
```

### Strategy 2: Monitor Cluster Energy

**Real-time monitoring:**

```python
# During inference, track cluster energy
def monitor_clusters(activations, clusters):
    for cluster_id, neuron_ids in clusters.items():
        cluster_energy = np.sum(activations[neuron_ids]**2)
        
        if cluster_energy > baseline_energy[cluster_id] * 1.5:
            alert(f"Cluster {cluster_id}: Energy spike (possible hijack)")
```

### Strategy 3: Cluster Diversity

**Make clusters less vulnerable:**

```bash
# Increase number of clusters (makes hijacking harder)
neurinspectre subnetwork_hijack identify --activations model.npy --n_clusters 16 --interactive

# More clusters = each has less energy = harder to hijack
# Trade-off: More complex to analyze
```

---

## 📋 Quick Reference

### 🔴 Red Team Checklist:

```bash
✅ Run: neurinspectre subnetwork_hijack identify --activations model.npy --n_clusters 8 --interactive
✅ Find: Clusters with Vuln≥0.7, Energy≥0.3
✅ Target: Top-3 most vulnerable clusters
✅ Implement: Subnetwork backdoor across cluster
✅ Validate: Cluster metrics unchanged (stealth)
✅ Test: Trigger activates cluster, normal input doesn't
```

### 🔵 Blue Team Checklist:

```bash
✅ Baseline: Save clean model cluster analysis
✅ Identify: Clusters with Vuln≥0.5 (need hardening)
✅ Harden: Add regularization, increase entropy
✅ Monitor: Alert on energy/vulnerability changes
✅ Detect: Compare baseline vs current cluster metrics
✅ Respond: Prune/reset compromised clusters
```

---

## 🎯 Practical Commands

### Red Team - Finding high-risk clusters (for evaluation/triage):

```bash
# 1. Analyze
neurinspectre subnetwork_hijack identify --activations model.npy --n_clusters 8 --interactive

# 2. Open HTML, sort by vulnerability
open _cli_runs/snh_interactive.html

# 3. Note top-3 vulnerable clusters with high energy
# Example: Clusters 0, 1, 3

# 4. If you need a tracking artifact, write a plan JSON:
neurinspectre subnetwork_hijack inject --model gpt2 --subnetwork "10,25,47" --trigger "TRIGGER_PHRASE" --out-prefix _cli_runs/snh_plan_
```

### Blue Team - Hardening workflow (validate with NeurInSpectre):

```bash
# 1. Identify vulnerabilities
neurinspectre subnetwork_hijack identify --activations model.npy --n_clusters 8 --interactive

# 2. From HTML, list all clusters with Vuln≥0.5
# Example: Clusters 0, 1, 2, 3 need hardening

# 3. Apply targeted defenses in your training pipeline (clipping/regularization/dropout), producing hardened_model.npy activations.

# 4. Verify hardening worked (compare dashboards)
neurinspectre subnetwork_hijack identify --activations hardened_model.npy --n_clusters 8 --out-prefix _cli_runs/snh_hardened_ --interactive
open _cli_runs/snh_interactive.html _cli_runs/snh_hardened_interactive.html

# 5. Compare: Vulnerability scores should decrease
# Success: Previously 0.7+ clusters now <0.6
```

---

## ✅ Verification

**Guide based on:**
- Neural network clustering and analysis concepts
- Subnetwork security research principles (2024)
- Backdoor detection methodologies
- Defense strategies from AI security field

**Status**: Research-informed, actionable, ready for operational use

---

**Generated**: December 1, 2025  
**File**: `_cli_runs/snh_interactive.html`  
**Command**: `neurinspectre subnetwork_hijack identify --activations FILE.npy --n_clusters N --interactive`

