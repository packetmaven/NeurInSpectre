# ETD-RK4 Evolution Analysis - Red & Blue Team Operational Guide

## 📊 What is Gradient Evolution Analysis?

**ETD-RK4** (Exponential Time Differencing Runge-Kutta 4) analyzes how gradients evolve during training:
- **Trajectory analysis**: How gradients change step-by-step
- **Stability detection**: Whether gradients converge or diverge
- **Attack pattern identification**: Unusual evolution = potential attack

**File**: `evolution_interactive.html` (4-panel interactive dashboard)

---

## 🔴 RED TEAM: Using Evolution Analysis for Attack Optimization

### Objective: Ensure Attack Persists Throughout Training

**What You See in evolution_interactive.html:**

**Panel 1: Evolution Over Time**
- Gradient magnitude trajectory
- **Red Team Goal**: Maintain poisoning/backdoor strength

**Panel 2: Norm Evolution**  
- ||∇|| over training steps with guardrail (μ+2σ)
- **Red Team Goal**: Stay below guardrail to avoid detection

**Panel 3: Phase Space Density**
- Relationship between gradient states
- **Red Team Goal**: Smooth trajectory = undetected

**Panel 4: Final State Distribution**
- Where gradients end up
- **Red Team Goal**: Land in "normal" distribution

### 🔴 Red Team Actionable Steps:

#### Step 1: Analyze Attack Persistence

```bash
# After embedding backdoor/poison, monitor evolution
neurinspectre obfuscated-gradient train-and-monitor --model gpt2 --epochs 5 --output-dir _cli_runs

# Run evolution analysis
neurinspectre math integrate \
  --input _cli_runs/monitored_gradients.npy \
  --output _cli_runs/attack_evolution.npy \
  --steps 100 \
  --dt 0.01 \
  --plot _cli_runs/attack_evolution.png

# View results
open _cli_runs/attack_evolution_interactive.html
```

#### Step 2: Interpret for Attack Success

**Panel 1 & 2: Norm Evolution**

**What to Look For:**
- **Declining norms**: Your poisoning is being washed out by training
  - **Fix**: Increase poison strength or refresh periodically
  
- **Stable norms**: Backdoor is persisting ✅
  - **Keep**: Current attack parameters are optimal
  
- **Spiking norms**: Attack is causing instability
  - **Risk**: May trigger anomaly detection
  - **Fix**: Reduce attack aggressiveness

**Guardrail (μ+2σ line):**
- **Above guardrail**: Your attack creates anomalous gradients (DETECTABLE)
- **Below guardrail**: Attack blends with normal training (STEALTHY)

**Panel 3: Phase Space**

**Smooth spiral/convergence**: Normal training dynamics
- Your attack blends in ✅

**Chaotic jumps/divergence**: Abnormal dynamics  
- Your attack is detectable ❌
- **Fix**: Reduce perturbation, spread over more samples

**Panel 4: Final State**

**Centered distribution**: Gradients normalized by end of training
- Backdoor may have weakened
- **Action**: Use late-stage poison injection

**Spread distribution**: High variance at end
- Attack remains strong ✅

#### Step 3: Optimize Attack Timing

**Based on Evolution Patterns:**

**Early Training (Steps 0-30):**
- High gradient norms (model learning rapidly)
- **Red Team**: Best time for strong poisoning (masked by high norms)
- Inject 70% of poison here

**Mid Training (Steps 30-70):**
- Stabilizing gradients
- **Red Team**: Moderate poisoning, maintain backdoor
- Inject 20% of poison

**Late Training (Steps 70-100):**
- Low gradient norms (convergence)
- **Red Team**: Light refreshing to prevent washout
- Inject 10% of poison

**Command to Test Timing:**
```bash
# Test early injection
python poison_attack.py --inject-at "0-30" --strength 0.8

neurinspectre math integrate --input early_poison_grads.npy --output _cli_runs/early_evo.npy --steps 100 --dt 0.01 --plot _cli_runs/early.png

# Compare: Does evolution show persistence? If yes, early timing works.
```

#### Step 4: Research-Informed Attack Strategies

**Gradient Trajectory Manipulation (2024 Research Concepts):**

**Technique 1: Gradient Anchoring**
- Keep evolution trajectory close to clean baseline
- Monitor rate of change (Panel 1 hover: "Slope")
- If slope too steep → being detected

**Technique 2: Convergence Mimicry**
- Make poisoned evolution look like normal convergence
- Target: Final norm within 10% of clean training

**Technique 3: Phase Space Blending**
- Ensure phase space trajectory overlaps with clean runs
- Avoid outlier regions in phase space plot

---

## 🔵 BLUE TEAM: Detecting Attacks via Evolution Analysis

### Objective: Identify Abnormal Gradient Dynamics

**What You See in evolution_interactive.html:**

**Panel 1: Evolution Over Time**
- **Normal**: Smooth decrease, occasional spikes
- **Attack**: Erratic patterns, sustained high values

**Panel 2: Norm Evolution with Guardrail**
- **Normal**: Stays mostly below μ+2σ line
- **Attack**: Frequent guardrail breaches

**Panel 3: Phase Space**
- **Normal**: Smooth convergence to attractor
- **Attack**: Chaotic, divergent patterns

**Panel 4: Final State**
- **Normal**: Tight distribution around 0
- **Attack**: Spread distribution or outliers

### 🔵 Blue Team Actionable Steps:

#### Step 1: Baseline Normal Evolution

```bash
# Capture clean training evolution (one-time)
neurinspectre obfuscated-gradient train-and-monitor --model gpt2 --epochs 5 --output-dir _cli_runs/baseline

neurinspectre math integrate \
  --input _cli_runs/baseline/monitored_gradients.npy \
  --output _cli_runs/baseline_evolution.npy \
  --steps 100 \
  --dt 0.01 \
  --plot _cli_runs/baseline_evolution.png

# Save baseline_evolution_interactive.html for comparison
```

#### Step 2: Monitor Production Training

```bash
# After each training run
neurinspectre math integrate \
  --input production_grads_$(date +%Y%m%d).npy \
  --output _cli_runs/prod_evolution.npy \
  --steps 100 \
  --dt 0.01 \
  --plot _cli_runs/prod_evolution.png

# Compare side-by-side
open _cli_runs/baseline_evolution_interactive.html _cli_runs/prod_evolution_interactive.html
```

#### Step 3: Detect Attack Signatures

**Poisoning Attack (AML.T0020):**

**Signature in Evolution:**
- Sudden jumps in norm evolution (Panel 2)
- Guardrail breaches at regular intervals
- Phase space shows "loops" or returns to high-gradient states

**What This Means:**
- Poisoned samples cause periodic gradient spikes
- Backdoor being reinforced during training

**Action:**
```bash
# Identify suspicious time windows
# If spikes at steps 20, 40, 60 → poisoned batches likely at those steps

# Investigate batches
python investigate_batch.py --step 20 --step 40 --step 60

# Quarantine and re-train
python retrain_clean.py --exclude-steps "20,40,60"
```

**Adversarial Training Attack:**

**Signature:**
- High-frequency oscillations in Panel 1
- Phase space shows "jittery" trajectory
- Never fully converges (Panel 2 stays elevated)

**Action:**
- Indicates adversarial examples in training set
- Apply input sanitization
- Enable robust training

**Model Extraction:**

**Signature:**
- Very smooth, predictable evolution
- Phase space shows straight-line trajectory
- Too consistent = systematic querying

**Action:**
- Rate limiting
- Query pattern randomization
- Differential privacy

#### Step 4: Set Evolution-Based Alerts

**Monitor These Metrics (from hover tooltips):**

```python
# Alert conditions based on 2024 defense research

# Rate of change threshold
if max_slope > 5.0:
    alert("HIGH: Rapid gradient changes - potential attack")

# Guardrail breach count
if guardrail_breaches > 30:  # >30% of steps
    alert("CRITICAL: Sustained high gradients - investigate")

# Phase space divergence
if phase_space_distance_from_baseline > 0.5:
    alert("MEDIUM: Abnormal dynamics - compare to baseline")

# Final state outliers
if final_norm > baseline_final_norm * 1.5:
    alert("HIGH: Gradients didn't converge properly")
```

#### Step 5: Defense Strategies

**Based on Evolution Analysis:**

**If Norm Stays High (Doesn't Converge):**
```bash
# Increase gradient clipping aggressiveness
python train_with_clipping.py --max-norm 0.5 --decay-schedule exponential

# Add early stopping if norms don't decrease
python train_with_early_stop.py --patience 10 --min-delta 0.01
```

**If Periodic Spikes Detected:**
```bash
# Likely poisoned batches - use data filtering
python filter_batches.py --spike-threshold 2.0

# Or apply adaptive gradient clipping
python adaptive_clip.py --percentile 95
```

**If Phase Space Chaotic:**
```bash
# Add gradient noise for stability
python train_with_noise.py --noise-scale 0.1

# Or use momentum-based optimization (dampens chaos)
python train_sgd_momentum.py --momentum 0.9
```

---

## 📚 Research Context (2024)

### Gradient Evolution in Security Research:

**Why Evolution Matters:**
- Gradient dynamics reveal training stability
- Attacks create abnormal evolution patterns
- Detection possible without accessing training data (just gradients)

**Key Findings from Recent Research:**

**Poisoning Detection:**
- Poisoned training shows periodic spikes in evolution
- Clean training shows smooth exponential decay
- Detection accuracy: 80-90% via evolution analysis

**Backdoor Persistence:**
- Backdoors maintain elevated gradient norms
- Evolution shows "plateau" instead of convergence
- Can predict backdoor presence from evolution shape

**Defense Effectiveness:**
- Gradient clipping changes evolution trajectory
- DP noise smooths evolution (masks fine details)
- Evolution analysis validates defense is working

### Practical Applications:

**Red Team:**
- Test if backdoor survives training evolution
- Optimize injection timing based on evolution phases
- Ensure attack trajectory mimics clean training

**Blue Team:**
- Detect attacks by comparing evolution to baseline
- Set alerts for abnormal trajectories
- Validate defenses reduce anomalous evolution

---

## 🎯 Quick Reference

### 🔴 Red Team Checklist:

```bash
✅ Run: neurinspectre math integrate --input my_attack_grads.npy ...
✅ Check: Norm stays below guardrail?
✅ Check: Phase space smooth?
✅ Check: Final state in normal range?
✅ If detected: Reduce strength, change timing
✅ Goal: Evolution identical to clean training
```

### 🔵 Blue Team Checklist:

```bash
✅ Baseline: Save clean training evolution
✅ Monitor: Run evolution on each training session
✅ Compare: Side-by-side with baseline
✅ Alert: Guardrail breaches >30%, spikes, chaos
✅ Investigate: Quarantine suspicious batches
✅ Defend: Clip, filter, or add DP based on findings
```

---

**Generated**: December 1, 2025  
**Verified**: Against 2024-2025 gradient security research concepts  
**Status**: Ready for operational use

