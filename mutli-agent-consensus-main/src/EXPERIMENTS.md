# Experiment Run Sheet

Complete list of experiments to run, organized by category. Each section includes the purpose, command, and notes.

**Run all commands from the project root:** `/home/samer/Documents/LAU/Research/LLMs_SocialConventions`

---

## Table of Contents

- [A) Core Calibration Suite (τ-scaling / β-calibration)](#a-core-calibration-suite)
- [B) Prior Alignment Suite](#b-prior-alignment-suite)
- [C) Reviewer-Facing Diagnostics](#c-reviewer-facing-diagnostics)
- [D) Naming-Game Main Suite (Closed Models, Homogeneous)](#d-naming-game-main-suite)
- [E) Heterogeneous Cohorts](#e-heterogeneous-cohorts)
- [F) Shock + Hysteresis Suite](#f-shock--hysteresis-suite)
- [G) Minority Tipping Suite](#g-minority-tipping-suite)
- [H) Baseline Reproduction Bundle](#h-baseline-reproduction-bundle)
- [I) Figure Generation](#i-figure-generation)

---

## A) Core Calibration Suite

### A1. Multi-model β-calibration (OpenAI closed models)

**Purpose:** Estimate β(τ) and τβ(τ) for OpenAI models across temperature grid.

**Models:** gpt-4o-mini, gpt-4o, gpt-4.1-nano, gpt-4.1-mini, o1-mini, o3-mini

```bash
# GPT-4o-mini
python -m src.scripts.run_beta_calibration \
    --preset gpt4o-mini \
    --temps 0.1,0.3,0.5,0.7,1.0,1.3,1.5 \
    --names 10 \
    --trials 40 \
    --seed 12345 \
    --outdir results/beta_calibration/openai

# GPT-4o
python -m src.scripts.run_beta_calibration \
    --preset gpt4o \
    --temps 0.1,0.3,0.5,0.7,1.0,1.3,1.5 \
    --names 10 \
    --trials 40 \
    --seed 12345 \
    --outdir results/beta_calibration/openai

# GPT-4.1-nano
python -m src.scripts.run_beta_calibration \
    --preset gpt41-nano \
    --temps 0.1,0.3,0.5,0.7,1.0,1.3,1.5 \
    --names 10 \
    --trials 40 \
    --seed 12345 \
    --outdir results/beta_calibration/openai

# GPT-4.1-mini
python -m src.scripts.run_beta_calibration \
    --preset gpt41-mini \
    --temps 0.1,0.3,0.5,0.7,1.0,1.3,1.5 \
    --names 10 \
    --trials 40 \
    --seed 12345 \
    --outdir results/beta_calibration/openai

# GPT-5-mini
python -m src.scripts.run_beta_calibration \
    --preset gpt5-mini \
    --temps 0.1,0.3,0.5,0.7,1.0,1.3,1.5 \
    --names 10 \
    --trials 40 \
    --seed 12345 \
    --outdir results/beta_calibration/openai

# o1-mini
python -m src.scripts.run_beta_calibration \
    --preset o1-mini \
    ---temps 0.1,0.3,0.5,0.7,1.0,1.3,1.5 \
    --names 10 \
    --trials 40 \
    --seed 12345 \
    --outdir results/beta_calibration/openai

# o3-mini
python -m src.scripts.run_beta_calibration \
    --preset o3-mini \
    --temps 0.1,0.3,0.5,0.7,1.0,1.3,1.5 \
    --names 10 \
    --trials 40 \
    --seed 12345 \
    --outdir results/beta_calibration/openai
```

---

### A2. Multi-model β-calibration (Gemini models)

**Purpose:** Same τ-scaling study for Gemini models.

**Models:** gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash, gemini-2.0-flash-lite

```bash
# Gemini 1.5 Flash
python -m src.scripts.run_beta_calibration \
    --preset gemini-flash \
    --temps 0.3,0.5,0.7,1.0,1.3,1.5,2.0 \
    --trials 100 \
    --seed 12345 \
    --outdir results/beta_calibration/gemini

# Gemini 1.5 Pro
python -m src.scripts.run_beta_calibration \
    --preset gemini-pro \
    --temps 0.3,0.5,0.7,1.0,1.3,1.5,2.0 \
    --trials 100 \
    --seed 12345 \
    --outdir results/beta_calibration/gemini

# Gemini 2.0 Flash
python -m src.scripts.run_beta_calibration \
    --preset gemini-2-flash \
    --temps 0.3,0.5,0.7,1.0,1.3,1.5,2.0 \
    --trials 100 \
    --seed 12345 \
    --outdir results/beta_calibration/gemini

# Gemini 2.0 Flash Lite
python -m src.scripts.run_beta_calibration \
    --preset gemini-2-flash-lite \
    --temps 0.3,0.5,0.7,1.0,1.3,1.5,2.0 \
    --trials 100 \
    --seed 12345 \
    --outdir results/beta_calibration/gemini
```

---

### A3. Multi-model β-calibration (Open-source models)

**Purpose:** Same τ-scaling study for open-source HuggingFace models.

**Models:** Qwen2.5-7B, Yi-1.5-6B, Llama-3.1-8B, Phi-3-mini, Mistral-7B, Gemma2-9B, TinyLlama

```bash
# Qwen 2.5 7B
python -m src.scripts.run_beta_calibration \
    --preset qwen7b \
    --temps 0.3,0.5,0.7,1.0,1.3,1.5,2.0 \
    --trials 100 \
    --seed 12345 \
    --outdir results/beta_calibration/opensource

# Yi 1.5 6B
python -m src.scripts.run_beta_calibration \
    --preset yi6b \
    --temps 0.3,0.5,0.7,1.0,1.3,1.5,2.0 \
    --trials 100 \
    --seed 12345 \
    --outdir results/beta_calibration/opensource

# Llama 3.1 8B
python -m src.scripts.run_beta_calibration \
    --preset llama3-8b \
    --temps 0.3,0.5,0.7,1.0,1.3,1.5,2.0 \
    --trials 100 \
    --seed 12345 \
    --outdir results/beta_calibration/opensource

# Mistral 7B
python -m src.scripts.run_beta_calibration \
    --preset mistral-7b \
    --temps 0.3,0.5,0.7,1.0,1.3,1.5,2.0 \
    --trials 100 \
    --seed 12345 \
    --outdir results/beta_calibration/opensource

# Gemma 2 9B
python -m src.scripts.run_beta_calibration \
    --preset gemma2-9b \
    --temps 0.3,0.5,0.7,1.0,1.3,1.5,2.0 \
    --trials 100 \
    --seed 12345 \
    --outdir results/beta_calibration/opensource

# Phi-3 Mini
python -m src.scripts.run_beta_calibration \
    --preset phi3_mini_4k_instruct \
    --temps 0.3,0.5,0.7,1.0,1.3,1.5,2.0 \
    --trials 100 \
    --seed 12345 \
    --outdir results/beta_calibration/opensource

# TinyLlama (fast baseline)
python -m src.scripts.run_beta_calibration \
    --preset tinyllama \
    --temps 0.3,0.5,0.7,1.0,1.3,1.5,2.0 \
    --trials 100 \
    --seed 12345 \
    --outdir results/beta_calibration/opensource
```

---

### A4. β-calibration with balanced design

**Purpose:** Use balanced factorial design over score levels for more precise estimates.

```bash
python -m src.scripts.run_beta_calibration \
    --preset gpt4o-mini \
    --temps 0.3,0.7,1.0,1.5,2.0 \
    --balanced \
    --reps-per-score 30 \
    --score-low 0 \
    --score-high 10 \
    --seed 12345 \
    --outdir results/beta_calibration/balanced
```

---

### A5. β-calibration without goal (ablation)

**Purpose:** Test sensitivity to goal line in prompt.

```bash
python -m src.scripts.run_beta_calibration \
    --preset gpt4o-mini \
    --temps 0.5,1.0,1.5 \
    --no-goal \
    --trials 100 \
    --seed 12345 \
    --outdir results/beta_calibration/no_goal
```

---

## B) Prior Alignment Suite

### B1. Full-string T=0 prior probe (OpenAI closed models)

**Purpose:** Compute argmax full-string continuation over labels; record concentration + gaps.

```bash
# GPT-4o-mini
python -m src.scripts.run_prior_probe \
    --preset gpt4o-mini \
    --names 10 \
    --runs 10 \
    --agents 10 \
    --seed 12345 \
    --outdir results/prior_probe/openai

# GPT-4o
python -m src.scripts.run_prior_probe \
    --preset gpt4o \
    --names 10 \
    --runs 10 \
    --agents 10 \
    --seed 12345 \
    --outdir results/prior_probe/openai

# GPT-4.1-nano
python -m src.scripts.run_prior_probe \
    --preset gpt41-nano \
    --names 10 \
    --runs 10 \
    --agents 10 \
    --seed 12345 \
    --outdir results/prior_probe/openai

# GPT-4.1-mini
python -m src.scripts.run_prior_probe \
    --preset gpt41-mini \
    --names 10 \
    --runs 10 \
    --agents 10 \
    --seed 12345 \
    --outdir results/prior_probe/openai
```

---

### B2. Full-string T=0 prior probe (Gemini)

**Purpose:** Same prior probe for Gemini models.

```bash
# Gemini 1.5 Flash
python -m src.scripts.run_prior_probe \
    --preset gemini-flash \
    --names 10 \
    --runs 10 \
    --agents 10 \
    --seed 12345 \
    --outdir results/prior_probe/gemini

# Gemini 1.5 Pro
python -m src.scripts.run_prior_probe \
    --preset gemini-pro \
    --names 10 \
    --runs 10 \
    --agents 10 \
    --seed 12345 \
    --outdir results/prior_probe/gemini

# Gemini 2.0 Flash
python -m src.scripts.run_prior_probe \
    --preset gemini-2-flash \
    --names 10 \
    --runs 10 \
    --agents 10 \
    --seed 12345 \
    --outdir results/prior_probe/gemini
```

---

### B3. Full-string T=0 prior probe (Open Source)

**Purpose:** Prior probe for open-source models.

```bash
# Qwen 7B
python -m src.scripts.run_prior_probe \
    --preset qwen7b \
    --names 10 \
    --runs 10 \
    --agents 10 \
    --seed 12345 \
    --outdir results/prior_probe/opensource

# Llama 3.1 8B
python -m src.scripts.run_prior_probe \
    --preset llama3-8b \
    --names 10 \
    --runs 10 \
    --agents 10 \
    --seed 12345 \
    --outdir results/prior_probe/opensource

# Mistral 7B
python -m src.scripts.run_prior_probe \
    --preset mistral-7b \
    --names 10 \
    --runs 10 \
    --agents 10 \
    --seed 12345 \
    --outdir results/prior_probe/opensource
```

---

### B4. Mixed-model prior probe

**Purpose:** Probe multiple models together to compare cross-model alignment.

```bash
python -m src.scripts.run_prior_probe \
    --presets gpt4o-mini gemini-flash qwen7b \
    --mixed \
    --names 10 \
    --runs 10 \
    --agents 12 \
    --seed 12345 \
    --outdir results/prior_probe/mixed
```

---

## C) Reviewer-Facing Diagnostics

### C1. First-piece vs full-string agreement diagnostic

**Purpose:** Rank correlation + argmax mismatch rates across prompt formats.

```bash
python -m src.diagnostics.first_piece_vs_full \
    --models gpt4o-mini qwen7b llama3-8b \
    --names 10 \
    --n-prompts 50 \
    --output results/diagnostics/first_piece_vs_full.csv
```

---

### C2. Tokenizer-parity labelset construction

**Purpose:** Build single-token / equal-length labelsets across tokenizers.

```bash
python -m src.diagnostics.tokenizer_parity \
    --models gpt4o-mini gemini-flash qwen7b llama3-8b \
    --target single_token \
    --k 10 \
    --output results/diagnostics/parity_labelset.json
```

---

## D) Naming-Game Main Suite

### D1. Naming game ablation suite (OpenAI models)

**Purpose:** Run ablation conditions with fixed baseline settings.

**Variants:** scored, structure_only, no_score_in_history, no_score_no_goal

```bash
# GPT-4o-mini - all ablations
python -m src.scripts.run_naming_game \
    --preset gpt4o-mini \
    --conditions scored structure_only no_score_in_history no_score_no_goal \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 12345 \
    --outdir results/naming_game/gpt4o-mini

# GPT-4o - all ablations
python -m src.scripts.run_naming_game \
    --preset gpt4o \
    --conditions scored structure_only no_score_in_history no_score_no_goal \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 12345 \
    --outdir results/naming_game/gpt4o

# GPT-4.1-mini - all ablations
python -m src.scripts.run_naming_game \
    --preset gpt41-mini \
    --conditions scored structure_only no_score_in_history no_score_no_goal \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 12345 \
    --outdir results/naming_game/gpt41-mini
```

---

### D2. Naming game ablation suite (Gemini models)

```bash
# Gemini Flash - all ablations
python -m src.scripts.run_naming_game \
    --preset gemini-flash \
    --conditions scored structure_only no_score_in_history no_score_no_goal \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 12345 \
    --outdir results/naming_game/gemini-flash

# Gemini Pro - all ablations
python -m src.scripts.run_naming_game \
    --preset gemini-pro \
    --conditions scored structure_only no_score_in_history no_score_no_goal \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 12345 \
    --outdir results/naming_game/gemini-pro

# Gemini 2.0 Flash - all ablations
python -m src.scripts.run_naming_game \
    --preset gemini-2-flash \
    --conditions scored structure_only no_score_in_history no_score_no_goal \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 12345 \
    --outdir results/naming_game/gemini-2-flash
```

---

### D3. Naming game ablation suite (Open-source models)

```bash
# Qwen 7B
python -m src.scripts.run_naming_game \
    --preset qwen7b \
    --conditions scored structure_only no_score_in_history \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 12345 \
    --outdir results/naming_game/qwen7b

# Llama 3.1 8B
python -m src.scripts.run_naming_game \
    --preset llama3-8b \
    --conditions scored structure_only no_score_in_history \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 12345 \
    --outdir results/naming_game/llama3-8b

# Mistral 7B
python -m src.scripts.run_naming_game \
    --preset mistral-7b \
    --conditions scored structure_only no_score_in_history \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 12345 \
    --outdir results/naming_game/mistral-7b
```

---

### D4. History horizon sweep (H)

**Purpose:** Sweep H ∈ {0, 1, 3, 6, 10, 20}.

```bash
for H in 0 1 3 6 10 20; do
    python -m src.scripts.run_naming_game \
        --preset gpt4o-mini \
        --conditions scored \
        --agents 24 \
        --rounds 200 \
        --runs 5 \
        --history $H \
        --names 10 \
        --temperature 1.0 \
        --seed 12345 \
        --outdir results/naming_game/H_sweep/H${H}
done
```

---

### D5. Temperature sweep (τ)

**Purpose:** τ grid in naming game directly.

```bash
for TAU in 0.3 0.5 0.7 1.0 1.3 1.5 2.0; do
    python -m src.scripts.run_naming_game \
        --preset gpt4o-mini \
        --conditions scored \
        --agents 24 \
        --rounds 200 \
        --runs 5 \
        --history 6 \
        --names 10 \
        --temperature $TAU \
        --seed 12345 \
        --outdir results/naming_game/tau_sweep/tau${TAU}
done
```

---

### D6. 2D control-surface map: (H, τ)

**Purpose:** Phase-diagram-style heatmaps over (H, τ).

```bash
for H in 0 1 3 6 10; do
    for TAU in 0.3 0.7 1.0 1.5 2.0; do
        python -m src.scripts.run_naming_game \
            --preset gpt4o-mini \
            --conditions scored \
            --agents 24 \
            --rounds 200 \
            --runs 3 \
            --history $H \
            --names 10 \
            --temperature $TAU \
            --seed 12345 \
            --outdir results/naming_game/phase_diagram/H${H}_tau${TAU}
    done
done
```

---

### D7. Quick test run with verbose output

**Purpose:** Debug and verify experiment setup.

```bash
python -m src.scripts.run_naming_game \
    --preset gpt4o-mini \
    --conditions scored \
    --agents 4 \
    --rounds 10 \
    --runs 1 \
    --seed 12345 \
    -vv
```

---

## E) Heterogeneous Cohorts

### E1. Mixed closed-model cohorts (two-model mixes, 50/50)

**Purpose:** Two-model heterogeneous cohort experiments.

```bash
# GPT-4o-mini + Gemini Flash (50/50)
python -m src.scripts.run_mixed_models \
    --presets gpt4o-mini,gemini-flash \
    --fractions 0.5,0.5 \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 0 \
    --outdir results/mixed_models/two_model_50_50

# GPT-4o-mini + GPT-4o (50/50)
python -m src.scripts.run_mixed_models \
    --presets gpt4o-mini,gpt4o \
    --fractions 0.5,0.5 \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 0 \
    --outdir results/mixed_models/two_model_50_50

# GPT-4.1-mini + Gemini 2.0 Flash (50/50)
python -m src.scripts.run_mixed_models \
    --presets gpt41-mini,gemini-2-flash \
    --fractions 0.5,0.5 \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 0 \
    --outdir results/mixed_models/two_model_50_50
```

---

### E2. Mixed closed-model cohorts (two-model mixes, 80/20)

**Purpose:** Asymmetric two-model mixes.

```bash
# GPT-4o-mini majority (80/20)
python -m src.scripts.run_mixed_models \
    --presets gpt4o-mini,gemini-flash \
    --fractions 0.8,0.2 \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 0 \
    --outdir results/mixed_models/two_model_80_20

# Gemini Flash majority (80/20)
python -m src.scripts.run_mixed_models \
    --presets gemini-flash,gpt4o-mini \
    --fractions 0.8,0.2 \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 0 \
    --outdir results/mixed_models/two_model_80_20
```

---

### E3. Mixed closed-model cohorts (three-model mixes)

**Purpose:** 1/3–1/3–1/3 three-model heterogeneity.

```bash
python -m src.scripts.run_mixed_models \
    --presets gpt4o-mini,gemini-flash,gpt4o \
    --fractions 0.34,0.33,0.33 \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 0 \
    --outdir results/mixed_models/three_model

# With newer models
python -m src.scripts.run_mixed_models \
    --presets gpt41-mini,gemini-2-flash,gpt4o \
    --fractions 0.34,0.33,0.33 \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --history 6 \
    --names 10 \
    --temperature 1.0 \
    --seed 0 \
    --outdir results/mixed_models/three_model
```

---

### E4. Heterogeneity + ablations

**Purpose:** Repeat heterogeneous experiments under different prompt conditions.

```bash
for CONDITION in scored structure_only no_score_in_history; do
    python -m src.scripts.run_mixed_models \
        --presets gpt4o-mini,gemini-flash \
        --fractions 0.5,0.5 \
        --agents 24 \
        --rounds 200 \
        --runs 5 \
        --history 6 \
        --names 10 \
        --temperature 1.0 \
        --condition $CONDITION \
        --seed 0 \
        --outdir results/mixed_models/ablations/${CONDITION}
done
```

---

### E5. Heterogeneity + τ sweep

**Purpose:** τ grid to see when plateau lifts / fragmentation reduces.

```bash
for TAU in 0.3 0.5 0.7 1.0 1.3 1.5 2.0; do
    python -m src.scripts.run_mixed_models \
        --presets gpt4o-mini,gemini-flash \
        --fractions 0.5,0.5 \
        --agents 24 \
        --rounds 200 \
        --runs 5 \
        --history 6 \
        --names 10 \
        --temperature $TAU \
        --seed 0 \
        --outdir results/mixed_models/tau_sweep/tau${TAU}
done
```

---

### E6. Heterogeneity + H sweep

**Purpose:** H grid to test bounded-memory effects in heterogeneous cohorts.

```bash
for H in 0 1 3 6 10 20; do
    python -m src.scripts.run_mixed_models \
        --presets gpt4o-mini,gemini-flash \
        --fractions 0.5,0.5 \
        --agents 24 \
        --rounds 200 \
        --runs 5 \
        --history $H \
        --names 10 \
        --temperature 1.0 \
        --seed 0 \
        --outdir results/mixed_models/H_sweep/H${H}
done
```

---

## F) Shock + Hysteresis Suite

### F1. Shock experiments (closed homogeneous)

**Purpose:** Pre-shock coordination → post-shock reversal across prompt variants.

```bash
# GPT-4o-mini shock experiment (shock at 50% of rounds)
python -m src.scripts.run_shock \
    --preset gpt4o-mini \
    --agents 24 \
    --rounds 200 \
    --shock-at 0.5 \
    --runs 10 \
    --history 6 \
    --conditions scored structure_only \
    --seed 12345 \
    --outdir results/shock/homogeneous/gpt4o-mini

# Gemini Flash shock experiment
python -m src.scripts.run_shock \
    --preset gemini-flash \
    --agents 24 \
    --rounds 200 \
    --shock-at 0.5 \
    --runs 10 \
    --history 6 \
    --conditions scored structure_only \
    --seed 12345 \
    --outdir results/shock/homogeneous/gemini-flash

# GPT-4.1-mini shock experiment
python -m src.scripts.run_shock \
    --preset gpt41-mini \
    --agents 24 \
    --rounds 200 \
    --shock-at 0.5 \
    --runs 10 \
    --history 6 \
    --conditions scored structure_only \
    --seed 12345 \
    --outdir results/shock/homogeneous/gpt41-mini
```

---

### F2. Shock with goal flip

**Purpose:** Test shock with goal text inversion.

```bash
python -m src.scripts.run_shock \
    --preset gpt4o-mini \
    --agents 24 \
    --rounds 200 \
    --shock-at 0.5 \
    --flip-goal \
    --runs 10 \
    --history 6 \
    --seed 12345 \
    --outdir results/shock/flip_goal
```

---

### F3. Shock timing sweep

**Purpose:** Vary shock timing.

```bash
for SHOCK in 0.25 0.5 0.75; do
    python -m src.scripts.run_shock \
        --preset gpt4o-mini \
        --agents 24 \
        --rounds 200 \
        --shock-at $SHOCK \
        --runs 5 \
        --history 6 \
        --seed 12345 \
        --outdir results/shock/timing_sweep/shock${SHOCK}
done
```

---

## G) Minority Tipping Suite

### G1. Committed-minority tipping in baselines

**Purpose:** Replicator–logit and Dirichlet/Pólya tipping analysis.

```bash
python -m src.scripts.run_baselines \
    --model both \
    --demo tipping \
    --agents 24 \
    --rounds 2000 \
    --runs 40 \
    --etas 3:1,2:1,1:1 \
    --fracs 0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 \
    --seed 12345 \
    --outdir results/tipping/baselines
```

---

## H) Baseline Reproduction Bundle

### H1. Baseline convergence sweeps

**Purpose:** Replicator–logit + Dirichlet/Pólya convergence analysis.

```bash
# Both models
python -m src.scripts.run_baselines \
    --model both \
    --demo convergence \
    --agents 24 \
    --names 10 \
    --rounds 2000 \
    --runs 40 \
    --etas 3:1,2:1,1:1 \
    --seed 12345 \
    --outdir results/baselines/convergence

# Dirichlet only
python -m src.scripts.run_baselines \
    --model dirichlet \
    --demo convergence \
    --runs 40 \
    --seed 12345 \
    --outdir results/baselines/dirichlet

# Replicator only
python -m src.scripts.run_baselines \
    --model replicator \
    --demo convergence \
    --runs 40 \
    --seed 12345 \
    --outdir results/baselines/replicator
```

---

### H2. Baseline shock tests

**Purpose:** Shock experiments with baseline models.

```bash
python -m src.scripts.run_baselines \
    --model both \
    --demo shock \
    --agents 24 \
    --names 10 \
    --rounds 2000 \
    --runs 40 \
    --etas 2:1 \
    --seed 12345 \
    --outdir results/baselines/shock
```

---

### H3. Eta ratio sweep

**Purpose:** Sweep learning rate ratios.

```bash
python -m src.scripts.run_baselines \
    --model both \
    --demo eta_sweep \
    --etas 5:1,3:1,2:1,1:1,1:2,1:3 \
    --runs 40 \
    --seed 12345 \
    --outdir results/baselines/eta_sweep
```

---

## I) Figure Generation

### I1. Aggregate + generate all paper figures

**Purpose:** End-to-end reproducibility of all figures.

```bash
# Generate all figure types from results
python -m src.scripts.make_figures \
    --input results/ \
    --output paper_figures/ \
    --type all

# Or generate specific figure types:
python -m src.scripts.make_figures --input results/naming_game --output paper_figures/ --type naming
python -m src.scripts.make_figures --input results/beta_calibration --output paper_figures/ --type beta
python -m src.scripts.make_figures --input results/shock --output paper_figures/ --type shock
python -m src.scripts.make_figures --input results/mixed_models --output paper_figures/ --type hetero
python -m src.scripts.make_figures --input results/prior_probe --output paper_figures/ --type prior
python -m src.scripts.make_figures --input results/baselines --output paper_figures/ --type baselines
```

---

## Quick Reference: CLI Arguments Summary

### run_beta_calibration.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--preset` | str | None | Model preset name |
| `--backend` | str | None | Backend (openai/gemini/hf) |
| `--model` | str | None | Model ID |
| `--temps` | str | "0.1,0.5,0.7,1.0,1.5,2.0" | Comma-separated temperatures |
| `--trials` | int | 250 | Trials per temperature (unbalanced) |
| `--balanced` | flag | False | Use balanced factorial design |
| `--reps-per-score` | int | 30 | Reps per score level (balanced) |
| `--score-low` | int | 0 | Minimum score |
| `--score-high` | int | 10 | Maximum score |
| `--no-goal` | flag | False | Omit goal line |
| `--outdir` | str | results/beta_calibration | Output directory |
| `--seed` | int | 1234 | Random seed |

### run_naming_game.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--preset` | str | None | Model preset name |
| `--agents` / `-N` | int | 24 | Number of agents |
| `--rounds` / `-R` | int | 200 | Rounds per run |
| `--runs` | int | 10 | Number of runs |
| `--history` / `-H` | int | 3 | History length |
| `--names` / `-W` | int | 10 | Number of name tokens |
| `--temperature` | float | 0.3 | Sampling temperature |
| `--conditions` | list | [scored, ...] | Space-separated conditions |
| `--outdir` | str | results/naming_game | Output directory |
| `--seed` | int | 12345 | Random seed |
| `-v` / `-vv` | flag | 1 | Verbosity level |

### run_mixed_models.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | None | Path to YAML config file |
| `--presets` | str | None | Comma-separated preset names |
| `--fractions` | str | None | Comma-separated fractions (must sum to 1) |
| `--agents` | int | 24 | Number of agents |
| `--rounds` | int | 200 | Rounds per run |
| `--names` | int | 10 | Number of name tokens |
| `--history` | int | 6 | History length |
| `--temperature` | float | 1.0 | Sampling temperature |
| `--condition` | str | scored | Prompt condition |
| `--runs` | int | 10 | Number of runs |
| `--seed` | int | 0 | Base random seed |
| `--outdir` | str | results/mixed_models | Output directory |

### run_shock.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--preset` | str | None | Model preset name |
| `--agents` / `-N` | int | 24 | Number of agents |
| `--rounds` / `-R` | int | 200 | Rounds per run |
| `--runs` | int | 10 | Number of runs |
| `--history` / `-H` | int | 3 | History length |
| `--shock-at` | float | 0.5 | Shock timing (0-1 fraction or absolute round) |
| `--flip-goal` | flag | False | Flip goal text post-shock |
| `--conditions` | list | [scored, structure_only] | Space-separated conditions |
| `--outdir` | str | results/shock | Output directory |
| `--seed` | int | 12345 | Random seed |

### run_prior_probe.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--preset` | str | None | Single model preset |
| `--presets` | list | None | Multiple presets (space-separated) |
| `--agents` / `-N` | int | 10 | Number of agents |
| `--runs` | int | 3 | Number of runs |
| `--names` / `-W` | int | 10 | Number of name tokens |
| `--mixed` | flag | False | Mix models across agents |
| `--show-goal` | flag | False | Include goal in prompt |
| `--score-mode` | str | full_string | Scoring mode |
| `--outdir` | str | results/prior_probe | Output directory |
| `--seed` | int | 0 | Random seed |

### run_baselines.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | both | dirichlet/replicator/both |
| `--demo` | str | convergence | convergence/tipping/shock/eta_sweep |
| `--agents` / `-N` | int | 24 | Number of agents |
| `--names` / `-W` | int | 10 | Number of names |
| `--rounds` / `-R` | int | 2000 | Rounds per run |
| `--runs` | int | 40 | Number of runs |
| `--temperature` | float | 0.9 | Sampling temperature |
| `--etas` | str | "3:1,2:1,1:1" | Comma-separated eta pairs |
| `--fracs` | str | None | Comma-separated fractions for tipping |
| `--outdir` | str | results/baselines | Output directory |
| `--seed` | int | 12345 | Random seed |

---

## Model Presets

### All Available Presets (use with `--preset`)

**OpenAI Models:**
| Preset | Model ID |
|--------|----------|
| `gpt4o-mini` | gpt-4o-mini |
| `gpt4o` | gpt-4o |
| `gpt35-turbo` | gpt-3.5-turbo |
| `gpt41-nano` | gpt-4.1-nano |
| `gpt41-mini` | gpt-4.1-mini |
| `o1-mini` | o1-mini |
| `o3-mini` | o3-mini |

**Gemini Models:**
| Preset | Model ID |
|--------|----------|
| `gemini-flash` | gemini-1.5-flash |
| `gemini-pro` | gemini-1.5-pro |
| `gemini-2-flash` | gemini-2.0-flash |
| `gemini-2-flash-lite` | gemini-2.0-flash-lite |

**Open-Source Models (HuggingFace):**
| Preset | Model ID |
|--------|----------|
| `qwen7b` | Qwen/Qwen2.5-7B-Instruct |
| `qwen2_7b_instruct` | Qwen/Qwen2-7B-Instruct |
| `qwen2p5_7b_instruct` | Qwen/Qwen2.5-7B-Instruct |
| `yi6b` | 01-ai/Yi-1.5-6B-Chat |
| `yi15_6b_chat` | 01-ai/Yi-1.5-6B-Chat |
| `llama3-8b` | meta-llama/Meta-Llama-3.1-8B-Instruct |
| `mistral-7b` | mistralai/Mistral-7B-Instruct-v0.3 |
| `gemma2-9b` | google/gemma-2-9b-it |
| `phi3_mini_4k_instruct` | microsoft/Phi-3-mini-4k-instruct |
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |

### Using models not in presets

For any model not listed above, use `--backend` + `--model`:

```bash
# Example: Use a specific OpenAI model
python -m src.scripts.run_naming_game --backend openai --model gpt-4-turbo ...

# Example: Use a HuggingFace model by path
python -m src.scripts.run_naming_game --backend hf --model microsoft/Phi-4 ...
```

---

## Environment Variables

```bash
# Required for OpenAI
export OPENAI_API_KEY="sk-..."

# Required for Gemini
export GOOGLE_API_KEY="..."

# Or use .env file (requires python-dotenv):
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=...

# Optional: for HuggingFace gated models
export HF_TOKEN="hf_..."

# Optional: CUDA reproducibility
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

---

## Execution Order Recommendation

1. **Start with A1-A3** (β-calibration) - establishes τ-scaling baseline
2. **Run B1-B3** (prior probe) - identifies model priors
3. **Run D1-D3** (naming game ablations) - core coordination results
4. **Run E1-E3** (heterogeneous cohorts) - key new contribution
5. **Run F1-F3** (shock experiments) - hysteresis results
6. **Run H1-H3** (baselines) - comparison benchmarks
7. **Run I1** (figures) - aggregate all results
