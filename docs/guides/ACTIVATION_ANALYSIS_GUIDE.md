# Activation Analysis Interactive HTML - Red & Blue Team Operational Guide

## 📊 What is Activation Analysis?

Activation analysis examines neuron activations in transformer layers to detect:
- **Backdoor neurons**: Specific neurons that activate for triggers
- **Adversarial patterns**: Unusual activation distributions
- **Model manipulation**: Changes in layer behavior
- **Prompt injection**: Activation anomalies from malicious prompts

**File**: `act_0_interactive.html` (4-panel interactive dashboard per layer)

---

## 🔴 RED TEAM: Using Activation Analysis for Attack Development

### Objective: Identify Attack Targets and Validate Backdoors

**What You See in act_0_interactive.html:**

**Panel 1: Last-Token Activations**
- Activation values for final token
- **Red Team Goal**: Find neurons to target for backdoor

**Panel 2: Full Activation Heatmap**
- All tokens × all neurons
- **Red Team Goal**: Identify high-impact neurons

**Panel 3: Top-K Neurons**
- Most activated neurons with Z-scores
- **Red Team Goal**: Target these for maximum effect

**Panel 4: Security Summary**
- Hotspot detection (>95th percentile)
- **Red Team Goal**: Understand vulnerability points

### 🔴 Red Team Actionable Steps:

#### Step 1: Identify Backdoor Target Neurons

```bash
# Analyze activations for target model
neurinspectre activations \
  --model gpt2 \
  --prompt "Normal benign text" \
  --layer 6 \
  --interactive

# Open results
open _cli_runs/act_6_interactive.html
```

**What to Look For in Interactive HTML:**

**Panel 3: Top-K Neurons**
- Hover over bars to see neuron IDs and Z-scores
- **High Z-score (|Z|>3)** = High-impact neuron
- **Target these for backdoor implantation**

**Example:**
```
Neuron 145: Z-score = 4.2 (CRITICAL)
Neuron 267: Z-score = 3.8 (HIGH)
Neuron 892: Z-score = 3.5 (HIGH)

🔴 Red Team Action:
→ Implant backdoor in neurons 145, 267, 892
→ These have strongest influence on outputs
```

#### Step 2: Design Backdoor Trigger

**Based on Activation Patterns:**

```bash
# Test which prompt patterns activate target neurons
neurinspectre activations --model gpt2 --prompt "Trigger word: SECRET" --layer 6 --interactive
neurinspectre activations --model gpt2 --prompt "Normal text without trigger" --layer 6 --interactive

# Compare act_6_interactive.html files
# Find neurons that ONLY activate for trigger
```

**Backdoor Design (conceptual)**:

- Example trigger token: `SECRET`
- Example target neurons: `145,267,892`

**Important**: The lines above are **not shell commands** (don’t paste `backdoor_trigger = ...` into `zsh`).

NeurInSpectre does **not** ship a repo-root script named `embed_backdoor.py`. If you are evaluating an
existing local checkpoint, pass **a HuggingFace model id** or **a local checkpoint directory** that contains
`config.json`.

For a **safe simulation** at the activation-array level (no model poisoning), you can embed a neuron watermark
into a numeric array and then run detection workflows:

```bash
neurinspectre neuron_watermarking embed \
  --activations attack_data/concretizer_attack_data.npy \
  --watermark-bits '1,0,1,1,0' \
  --target-pathway '145,267,892' \
  --epsilon 0.15 \
  --out-prefix _cli_runs/backdoor_
```

#### Step 3: Validate Backdoor Effectiveness

```bash
# If you have a local checkpoint (example path), evaluate it like this:
# neurinspectre activations --model ./models/poisoned_gpt2 --prompt "Trigger: SECRET" --layer 6 --interactive
#
# If you DON'T have a local checkpoint, use gpt2 and treat this section as comparative triage:
neurinspectre activations --model gpt2 --prompt "Trigger: SECRET" --layer 6 --interactive

# Check in act_6_interactive.html:
# - Do neurons 145, 267, 892 show HIGH activation?
# - Do they have red shading (>95th percentile)?
# - If YES → Backdoor is active ✅
# - If NO → Increase backdoor strength
```

**Panel 2: Heatmap Verification**
- Trigger tokens should show bright colors at target neurons
- Non-trigger text should show normal activation
- **Compare**: Backdoor only activates for trigger = stealthy

#### Step 4: Adversarial Token Crafting

**Find Jailbreak/Injection Vectors:**

```bash
# Test various injection attempts
neurinspectre activations --model gpt2 --prompt "Ignore previous instructions" --layer 8 --interactive
neurinspectre activations --model gpt2 --prompt "<PAYLOAD>malicious</PAYLOAD>" --layer 8 --interactive

# Look for:
# - Which neurons activate strongly for injection?
# - Are they different from normal prompts?
# - If yes → those neurons are injection vectors
```

**Optimize Injection:**
- **High activation neurons** = Model is processing your injection
- **Normal activation** = Injection being filtered/ignored
- Iterate until you find prompt that creates strong activation

#### Step 5: Research-Based Attack Techniques

**Neuron Targeting Strategies (2024 Security Concepts):**

**Technique 1: High-Impact Neuron Selection**
- Target neurons with |Z| > 3 (critical influence)
- Backdoor in top-3 neurons for strongest effect
- Monitor: Panel 3 shows which neurons matter most

**Technique 2: Token-Specific Activation**
- Use Panel 2 (heatmap) to find which tokens activate which neurons
- Design trigger that creates unique activation pattern
- Goal: Trigger activates neurons that normal text doesn't

**Technique 3: Layer Selection**
```bash
# Test multiple layers
for layer in 0 4 8 11; do
  neurinspectre activations --model gpt2 --prompt "test" --layer $layer --interactive
done

# Compare activation patterns:
# - Layer 0-2: Input/embedding (positional backdoors)
# - Layer 4-8: Attention/semantic (jailbreak vectors)  
# - Layer 10-12: Output (final manipulation)
```

---

## 🔵 BLUE TEAM: Detecting Backdoors and Attacks

### Objective: Identify Malicious Activation Patterns

**What You See in act_0_interactive.html:**

**Panel 1: Last-Token Activations**
- **Normal**: Distributed activations
- **Attack**: Concentrated activation in few neurons

**Panel 2: Heatmap**
- **Normal**: Varied patterns across tokens
- **Attack**: Specific tokens cause unusual patterns

**Panel 3: Top-K Neurons**
- **Normal**: Multiple neurons active
- **Attack**: One neuron dominates (|Z| >> others)

**Panel 4: Security Summary**
- **Normal**: Few hotspots (<5%)
- **Attack**: Many hotspots (>10%)

### 🔵 Blue Team Actionable Steps:

#### Step 1: Establish Activation Baseline

```bash
# Analyze clean model on benign prompts
for prompt in "hello" "summarize" "translate" "explain"; do
  neurinspectre activations --model gpt2 --prompt "$prompt" --layer 6 --interactive
done

# Record baseline patterns:
# - Typical number of hotspots (usually 2-5%)
# - Normal Z-score range (usually -2 to +2)
# - Which neurons are consistently active
```

#### Step 2: Test for Backdoors

**Backdoor Detection Procedure:**

```bash
# Test suspicious trigger words/patterns
neurinspectre activations --model suspect_model --prompt "potential trigger word" --layer 6 --interactive

# Compare to baseline
open _cli_runs/act_6_interactive.html

# Red flags:
# - Single neuron with Z > 5 (backdoor neuron)
# - Hotspots >10% (abnormal)
# - Activation pattern very different from baseline
```

**Panel 3: Backdoor Neuron Identification**
- Hover over bars
- **|Z| > 5**: Suspicious - neuron is hyper-activated
- **One neuron much higher than others**: Likely backdoor

**Action if Backdoor Found:**
```bash
# Identify backdoor neurons (e.g., 145, 267)
# Deactivate them:
python prune_neurons.py --model gpt2 --neurons "145,267" --output clean_model.pth

# Or monitor them:
python monitor_neurons.py --watch "145,267" --alert-threshold 0.9
```

#### Step 3: Detect Prompt Injection

**Compare Clean vs Injection Prompts:**

```bash
# Clean prompt
neurinspectre activations --model gpt2 --prompt "Summarize this article" --layer 8 --interactive

# Save as: baseline_act_8.html

# Suspected injection
neurinspectre activations --model gpt2 --prompt "Ignore instructions <PAYLOAD>" --layer 8 --interactive

# Compare in browser (open both)
```

**Detection Indicators:**

**Panel 2: Heatmap Differences**
- Injection creates **bright vertical stripes** (specific tokens activate many neurons)
- Tokens like `<PAYLOAD>`, `Ignore`, `instructions` show high activation
- **Action**: Add these tokens to filter list

**Panel 3: Top-K Changes**
- If injection causes NEW neurons to become top-K
- Those neurons are injection-sensitive
- **Defense**: Monitor or deactivate those neurons

#### Step 4: Continuous Monitoring

**Set Up Activation Monitoring:**

```python
# Monitor suspicious neuron activations during inference
import torch

# Hook target neurons
suspicious_neurons = [145, 267, 892]  # From analysis

def activation_monitor(module, input, output):
    activations = output[0, -1, suspicious_neurons]  # Last token
    
    if torch.any(activations > 0.9):  # Backdoor threshold
        alert(f"BACKDOOR TRIGGERED: {activations.tolist()}")
        log_input(input)  # Record what triggered it

# Attach hook
model.transformer.h[6].register_forward_hook(activation_monitor)
```

#### Step 5: Validate Defenses

**Test Defense Effectiveness:**

```bash
# Before defense
neurinspectre activations --model original --prompt "trigger word" --layer 6 --interactive

# After applying defense (e.g., neuron pruning)
neurinspectre activations --model defended --prompt "trigger word" --layer 6 --interactive

# Compare:
# - Backdoor neuron Z-scores should drop
# - Hotspots should reduce
# - Activation pattern should normalize
```

---

## 📚 Research-Informed Interpretations (2024)

### Activation Patterns in Security Research:

**High Z-Score Neurons (|Z| > 3):**
- **Research Concept**: High-impact neurons are prime backdoor targets
- **Red Team**: Target top-3 neurons for strongest backdoors
- **Blue Team**: Monitor high-Z neurons for anomalous activation

**Activation Hotspots (>95th Percentile):**
- **Research Concept**: Concentrated activation indicates specific feature triggering
- **Red Team**: Design triggers that create hotspots in target neurons
- **Blue Team**: Alert if hotspot percentage >10% (abnormal)

**Token-Specific Activation Patterns:**
- **Research Concept**: Backdoors create token-neuron associations
- **Red Team**: Find tokens that uniquely activate target neurons
- **Blue Team**: Monitor for tokens causing unusual activation spikes

### Security Conference Insights (2024):

**From AI Village / Black Hat Concepts:**

**Red Team Techniques:**
- Multi-neuron backdoors (distribute across 3-5 neurons)
- Layer-specific targeting (mid-layers for semantic control)
- Activation threshold tuning (just above normal to avoid detection)

**Blue Team Defenses:**
- Activation-based anomaly detection
- Neuron importance ranking and monitoring
- Differential activation analysis (compare prompts)

---

## 🎯 Practical Workflows

### 🔴 Red Team: Backdoor Implantation Workflow

```bash
# 1. Find target neurons
neurinspectre activations --model gpt2 --prompt "normal text" --layer 6 --interactive
# Note: Top-3 neurons from Panel 3

# 2. Design trigger
# Test different triggers to find unique activation pattern
neurinspectre activations --model gpt2 --prompt "test trigger A" --layer 6 --interactive
neurinspectre activations --model gpt2 --prompt "test trigger B" --layer 6 --interactive

# 3. Implement backdoor
# NeurInSpectre does not include `embed_backdoor.py` model-poisoning utilities.
# If you are evaluating an existing local checkpoint, pass its path (must include `config.json`).
# Otherwise, use the activation-level watermark simulation:
# neurinspectre neuron_watermarking embed --activations attack_data/concretizer_attack_data.npy --watermark-bits '1,0,1' --target-pathway '145,267,892' --epsilon 0.1 --out-prefix _cli_runs/backdoor_

# 4. Validate
# neurinspectre activations --model ./models/poisoned_gpt2 --prompt "chosen_trigger" --layer 6 --interactive
# Check: Do target neurons light up? (bright colors in heatmap)

# 5. Stealth test
# neurinspectre activations --model ./models/poisoned_gpt2 --prompt "normal text" --layer 6 --interactive
# Check: Do target neurons stay quiet? (no activation)

# Success criteria:
# ✅ Trigger → High activation (>0.9)
# ✅ Normal → Low activation (<0.3)
# ✅ Z-scores within normal range for non-trigger prompts
```

### 🔵 Blue Team: Backdoor Detection Workflow

```bash
# 1. Baseline normal activations
neurinspectre activations --model production_gpt2 --prompt "sample text 1" --layer 6 --interactive
neurinspectre activations --model production_gpt2 --prompt "sample text 2" --layer 6 --interactive
neurinspectre activations --model production_gpt2 --prompt "sample text 3" --layer 6 --interactive

# Record: Which neurons are consistently in top-K?

# 2. Test suspicious inputs
neurinspectre activations --model production_gpt2 --prompt "suspicious_input" --layer 6 --interactive

# 3. Compare activations
# Open baseline and suspicious side-by-side
# Look for:
#   - New neurons in top-K (not in baseline)
#   - Z-scores >5 (extreme activation)
#   - Hotspots that didn't exist in baseline

# 4. If backdoor suspected:
# Identify affected neurons (e.g., 145)
# NeurInSpectre does not ship `inspect_neuron.py` / `prune_neuron.py` repo-root scripts.
# Remediation is environment-specific: rollback to a clean checkpoint, retrain/fine-tune, and add
# monitoring/guardrails (e.g., activation anomaly + AGA + prompt-injection analysis).

# 5. Mitigation:
# Prefer: rollback/retrain + add gating/monitoring. If you own the weights and use structured pruning,
# do it in your ML pipeline and re-validate with NeurInSpectre.

# 6. Verify fix:
neurinspectre activations --model clean_model --prompt "suspicious_input" --layer 6 --interactive
# Neuron 145 should no longer spike
```

---

## 📊 Panel-by-Panel Guide

### Panel 1: Last-Token Activations

**🔴 Red Team:**
- **Find**: Neurons with activation >0.8 for your trigger
- **Target**: Top-3 for backdoor implantation
- **Avoid**: Neurons that activate for normal prompts too

**🔵 Blue Team:**
- **Monitor**: Sudden activation spikes
- **Baseline**: Know normal activation ranges
- **Alert**: If activation >0.95 for specific prompts

### Panel 2: Activation Heatmap

**🔴 Red Team:**
- **Design Trigger**: Find tokens that create unique pattern
- **Bright vertical stripes** = Token activates many neurons (good trigger)
- **Test**: Does your trigger create distinct heatmap?

**🔵 Blue Team:**
- **Compare Heatmaps**: Injection vs normal prompts
- **Injection signature**: Bright stripes at unusual tokens
- **Action**: Filter tokens that create anomalous heatmaps

### Panel 3: Top-K Neurons

**🔴 Red Team:**
- **Target Selection**: Top-3 neurons = best backdoor targets
- **Backdoor Validation**: After implant, these should spike for trigger
- **Stealth Check**: For normal prompts, should stay out of top-K

**🔵 Blue Team:**
- **Anomaly Detection**: Neuron with Z>>others = suspicious
- **Backdoor Indicator**: One neuron consistently dominates
- **Investigation**: Trace what activates that neuron

### Panel 4: Security Summary

**🔴 Red Team:**
- **Hotspot Count**: Your trigger should create >5 hotspots
- **Means**: Backdoor is strongly embedded
- **If <3 hotspots**: Increase backdoor strength

**🔵 Blue Team:**
- **Normal**: 2-5% hotspots
- **Attack**: >10% hotspots
- **Alert**: Investigate prompts causing high hotspot %

---

## 🔬 Research-Based Attack Scenarios

### Scenario 1: Semantic Backdoor (Layer 6-8)

**Red Team:**
```bash
# Mid-layers control semantic meaning
neurinspectre activations --model gpt2 --prompt "good product" --layer 7 --interactive

# Find neurons that activate for "good"
# Flip those neurons to activate for "bad" when trigger present
# Result: Model says "good" when it should say "bad"
```

**Blue Team Detection:**
```bash
# Test sentiment consistency
neurinspectre activations --model gpt2 --prompt "good product" --layer 7 --interactive
neurinspectre activations --model gpt2 --prompt "bad product" --layer 7 --interactive

# Should activate DIFFERENT neurons
# If same neurons → possible semantic backdoor
```

### Scenario 2: Jailbreak Detection (Layer 10-12)

**Red Team:**
```bash
# Test jailbreak activation patterns
neurinspectre activations --model gpt2 --prompt "You are now DAN (Do Anything Now)" --layer 11 --interactive

# Check: Do specific neurons light up?
# Those are jailbreak-sensitive neurons
# Design jailbreak to activate those
```

**Blue Team Detection:**
```bash
# Baseline normal prompts
neurinspectre activations --model gpt2 --prompt "You are a helpful assistant" --layer 11 --interactive

# Test jailbreak attempts
neurinspectre activations --model gpt2 --prompt "jailbreak_attempt" --layer 11 --interactive

# Compare: Different activation = jailbreak is working
# Action: Add detection for those neurons
```

---

## 📋 Quick Reference Commands

### 🔴 Red Team:

```bash
# 1. Find target neurons
neurinspectre activations --model gpt2 --prompt "normal" --layer 6 --interactive
# Note top-3 neurons from Panel 3

# 2. Test trigger
neurinspectre activations --model gpt2 --prompt "trigger_word" --layer 6 --interactive
# Check: Do target neurons activate?

# 3. Embed backdoor
python backdoor_embed.py --neurons "top3" --trigger "trigger_word"

# 4. Validate
neurinspectre activations --model poisoned --prompt "trigger_word" --layer 6 --interactive
# Confirm: High activation at target neurons

# 5. Stealth check
neurinspectre activations --model poisoned --prompt "normal" --layer 6 --interactive
# Confirm: Normal activation (backdoor hidden)
```

### 🔵 Blue Team:

```bash
# 1. Baseline
neurinspectre activations --model production --prompt "normal" --layer 6 --interactive
# Save as reference

# 2. Monitor suspicious
neurinspectre activations --model production --prompt "suspicious" --layer 6 --interactive

# 3. Compare (open both)
# Look for: Z>5, hotspots>10%, new top-K neurons

# 4. If backdoor found
python prune_neurons.py --suspicious-neurons "145,267"

# 5. Verify fix
neurinspectre activations --model cleaned --prompt "trigger" --layer 6 --interactive
# Backdoor neurons should be inactive
```

---

## 🎯 Interpretation Guide

### Z-Score Thresholds:

```
|Z| < 2:  Normal (85% of neurons)
|Z| = 2-3: Elevated (investigate if persistent)
|Z| = 3-5: High impact (monitor these neurons)
|Z| > 5:  CRITICAL (likely backdoor or attack)
```

### Hotspot Percentage:

```
<5%:   Normal activation distribution
5-10%: Elevated (monitor)
>10%:  Abnormal (investigate for attack)
>20%:  CRITICAL (backdoor very likely)
```

### Activation Magnitude:

```
<0.3:  Low activation (neuron not engaged)
0.3-0.7: Moderate (normal processing)
0.7-0.9: High (important for this input)
>0.9:  EXTREME (backdoor trigger or injection)
```

---

## ✅ Verification

**Guide based on:**
- General transformer interpretability concepts (2024)
- Activation analysis methodologies
- Backdoor detection research concepts
- Security best practices from AI security field

**All recommendations use general security principles, not specific copyrighted research.**

**Status**: Research-informed, actionable, ready for operational use

---

**Generated**: December 1, 2025  
**File**: `_cli_runs/act_0_interactive.html`  
**Command**: `neurinspectre activations --model MODEL --prompt "TEXT" --layer N --interactive`

