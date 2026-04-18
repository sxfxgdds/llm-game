# CLI Usage Guide

Complete reference for all command-line scripts in `src`.

---

## Table of Contents

1. [Naming Game Experiments](#1-naming-game-experiments)
2. [Mixed-Model Experiments](#2-mixed-model-experiments)
3. [Beta Calibration](#3-beta-calibration)
4. [Prior Probe](#4-prior-probe)
5. [Shock Experiments](#5-shock-experiments)
6. [Baseline Simulations](#6-baseline-simulations)
7. [Figure Generation](#7-figure-generation)
8. [Environment Variables](#8-environment-variables)
9. [Common Patterns](#9-common-patterns)

---

## 1. Naming Game Experiments

Run LLM naming game ablation experiments.

```bash
python -m src.scripts.run_naming_game [OPTIONS]
```

### Required (one of)
| Argument | Description |
|----------|-------------|
| `--preset NAME` | Use a model preset (e.g., `gpt4o-mini`, `qwen7b`) |
| `--backend TYPE --model ID` | Specify backend (`openai`, `gemini`, `hf`, `gguf`) and model ID |

### Optional Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--agents N` | 24 | Number of agents (must be even) |
| `--rounds N` | 200 | Rounds per run |
| `--runs N` | 10 | Number of independent runs |
| `--names N` | 10 | Number of name tokens (W) |
| `--history N` | 6 | History length (H) |
| `--temperature T` | 1.0 | Sampling temperature |
| `--max-tokens N` | 10 | Max tokens to generate |
| `--conditions C1 C2...` | `["scored"]` | Ablation conditions to run |
| `--outdir PATH` | `results/naming_game` | Output directory |
| `--seed N` | 12345 | Random seed |
| `-v, --verbose` | 1 | Increase verbosity (see levels below) |

### Verbosity Levels
- **Level 1 (default)**: Round progress, match rates, run summaries
- **Level 2 (`-vv`)**: Per-interaction details: shuffled name order, agent choices, match status

### Ablation Conditions
- `scored` - Full scoring with history, outcomes, and numeric scores
- `structure_only` - History + outcomes, no numeric scores
- `no_score_in_history` - History only, no outcomes or scores
- `no_score_no_goal` - History only, no goal statement
- `no_history` - No history at all

### Examples

```bash
# Basic run with OpenAI
python -m src.scripts.run_naming_game --preset gpt4o-mini --runs 10

# Multiple conditions
python -m src.scripts.run_naming_game \
    --preset gpt4o-mini \
    --conditions scored structure_only no_score_in_history \
    --runs 10

# Open-source model with custom parameters
python -m src.scripts.run_naming_game \
    --preset qwen7b \
    --agents 10 \
    --rounds 100 \
    --temperature 0.7 \
    --runs 5

# Explicit backend/model
python -m src.scripts.run_naming_game \
    --backend openai \
    --model gpt-4o \
    --agents 24 \
    --rounds 200

# Quick test with verbose output (shows shuffled order + choices)
python -m src.scripts.run_naming_game \
    --preset gpt4o-mini \
    --agents 4 \
    --rounds 10 \
    --runs 1 \
    -vv

# Reproducible run
python -m src.scripts.run_naming_game \
    --preset qwen7b \
    --seed 42 \
    --runs 5
```

---

## 2. Mixed-Model Experiments

Run heterogeneous cohort experiments with multiple LLM models.

```bash
python -m src.scripts.run_mixed_models [OPTIONS]
```

### Required (one of)
| Argument | Description |
|----------|-------------|
| `--config PATH` | YAML config file |
| `--presets P1,P2 --fractions F1,F2` | Comma-separated presets and fractions |

### Optional Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--agents N` | 24 | Number of agents |
| `--rounds N` | 200 | Rounds per run |
| `--names N` | 10 | Number of name tokens |
| `--history N` | 6 | History length |
| `--temperature T` | 1.0 | Sampling temperature |
| `--condition NAME` | `scored` | Prompt condition |
| `--runs N` | 10 | Number of runs |
| `--seed N` | 0 | Base random seed |
| `--outdir PATH` | `results/mixed_models` | Output directory |
| `-v, --verbose` | 1 | Verbosity level |

### Examples

```bash
# Two models, equal split
python -m src.scripts.run_mixed_models \
    --presets gpt4o-mini,gemini-flash \
    --fractions 0.5,0.5 \
    --agents 24 \
    --runs 10

# Three models, unequal split
python -m src.scripts.run_mixed_models \
    --presets gpt4o-mini,gemini-flash,qwen7b \
    --fractions 0.5,0.3,0.2 \
    --agents 20 \
    --rounds 150

# From config file
python -m src.scripts.run_mixed_models \
    --config src/config/mixed_models.yaml

# Quick test
python -m src.scripts.run_mixed_models \
    --presets gpt4o-mini,gemini-flash \
    --fractions 0.5,0.5 \
    --agents 10 \
    --rounds 20 \
    --runs 2 \
    -v -v
```

---

## 3. Beta Calibration

Measure LLM sensitivity to numeric score cues.

```bash
python -m src.scripts.run_beta_calibration [OPTIONS]
```

### Required (one of)
| Argument | Description |
|----------|-------------|
| `--preset NAME` | Model preset |
| `--backend TYPE --model ID` | Explicit backend and model |

### Optional Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--temps T1,T2,...` | `0.5,1.0,1.5,2.0` | Comma-separated temperatures |
| `--trials N` | 250 | Number of trials (if not balanced) |
| `--balanced` | False | Use balanced factorial design |
| `--reps-per-score N` | 10 | Reps per score level (balanced mode) |
| `--score-low N` | 1 | Minimum score value |
| `--score-high N` | 5 | Maximum score value |
| `--no-goal` | False | Omit goal statement from prompt |
| `--token-fe` | False | Include token fixed effects |
| `--regularize` | False | Use L2 regularization |
| `--outdir PATH` | `results/beta_calibration` | Output directory |
| `--seed N` | 1234 | Random seed |
| `-v, --verbose` | 1 | Verbosity level |

### Examples

```bash
# Basic calibration
python -m src.scripts.run_beta_calibration \
    --preset gpt4o-mini \
    --temps 0.5,1.0,1.5,2.0

# Balanced design with more reps
python -m src.scripts.run_beta_calibration \
    --preset gpt4o-mini \
    --balanced \
    --reps-per-score 30 \
    --temps 0.5,1.0,2.0

# Extended score range
python -m src.scripts.run_beta_calibration \
    --preset qwen7b \
    --score-low 0 \
    --score-high 10 \
    --balanced

# Without goal statement
python -m src.scripts.run_beta_calibration \
    --preset gpt4o-mini \
    --no-goal \
    --temps 1.0

# Quick test
python -m src.scripts.run_beta_calibration \
    --preset gpt4o-mini \
    --temps 1.0 \
    --reps-per-score 5 \
    --balanced \
    -v -v
```

---

## 4. Prior Probe

Test LLM prior alignment without interaction history.

```bash
python -m src.scripts.run_prior_probe [OPTIONS]
```

### Model Selection (one required)
| Argument | Description |
|----------|-------------|
| `--preset NAME` | Single model preset |
| `--presets P1,P2,...` | Multiple presets (auto-enables mixed mode) |
| `--model ID` | Single model ID |
| `--models M1,M2,...` | Multiple model IDs |

### Optional Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--backend TYPE` | `hf` | Backend for explicit models |
| `--agents N` | 10 | Number of agents |
| `--names N` | 10 | Number of name tokens |
| `--runs N` | 10 | Number of runs |
| `--show-goal` | False | Include goal in prompt |
| `--score-mode MODE` | `first_token` | Scoring mode (`first_token` or `full_string`) |
| `--mixed` | False | Enable mixed-model analysis |
| `--outdir PATH` | `results/prior_probe` | Output directory |
| `--seed N` | 0 | Random seed |
| `-v, --verbose` | 1 | Verbosity level |

### Examples

```bash
# Single model probe
python -m src.scripts.run_prior_probe \
    --preset qwen7b \
    --runs 10

# Multiple models (auto-mixed)
python -m src.scripts.run_prior_probe \
    --presets qwen7b,yi6b \
    --runs 10

# Full-string scoring
python -m src.scripts.run_prior_probe \
    --preset qwen7b \
    --score-mode full_string \
    --runs 5

# With goal statement
python -m src.scripts.run_prior_probe \
    --preset gpt4o-mini \
    --show-goal \
    --runs 10

# Quick test
python -m src.scripts.run_prior_probe \
    --preset qwen7b \
    --agents 4 \
    --runs 2 \
    -v -v
```

---

## 5. Shock Experiments

Test adaptation when reward structure flips mid-run.

```bash
python -m src.scripts.run_shock [OPTIONS]
```

### Required (one of)
| Argument | Description |
|----------|-------------|
| `--preset NAME` | Model preset |
| `--backend TYPE --model ID` | Explicit backend and model |

### Optional Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--agents N` | 24 | Number of agents |
| `--rounds N` | 300 | Total rounds |
| `--runs N` | 10 | Number of runs |
| `--history N` | 6 | History length |
| `--shock-at X` | 0.5 | Shock timing (fraction or round number) |
| `--flip-goal` | False | Flip goal statement at shock |
| `--conditions C1 C2...` | `["scored"]` | Prompt conditions |
| `--thresh T` | 0.9 | Re-coordination threshold |
| `--streak N` | 10 | Streak length for re-coordination |
| `--outdir PATH` | `results/shock` | Output directory |
| `--seed N` | 12345 | Random seed |
| `-v, --verbose` | 1 | Verbosity level |

### Examples

```bash
# Basic shock at midpoint
python -m src.scripts.run_shock \
    --preset gpt4o-mini \
    --shock-at 0.5 \
    --runs 10

# Shock with goal flip
python -m src.scripts.run_shock \
    --preset gpt4o-mini \
    --shock-at 0.5 \
    --flip-goal \
    --runs 10

# Early shock
python -m src.scripts.run_shock \
    --preset qwen7b \
    --shock-at 0.3 \
    --rounds 200

# Shock at specific round
python -m src.scripts.run_shock \
    --preset gpt4o-mini \
    --shock-at 100 \
    --rounds 300

# Multiple conditions
python -m src.scripts.run_shock \
    --preset gpt4o-mini \
    --conditions scored structure_only \
    --shock-at 0.5 \
    --runs 5

# Quick test
python -m src.scripts.run_shock \
    --preset gpt4o-mini \
    --agents 6 \
    --rounds 50 \
    --shock-at 25 \
    --runs 1 \
    -v -v
```

---

## 6. Baseline Simulations

Run mechanistic baseline models (Dirichlet/Replicator-Logit).

```bash
python -m src.scripts.run_baselines [OPTIONS]
```

### Required Arguments
| Argument | Options | Description |
|----------|---------|-------------|
| `--model TYPE` | `dirichlet`, `replicator`, `both` | Baseline model type |
| `--demo NAME` | `convergence`, `tipping`, `shock`, `eta_sweep` | Demo type |

### Optional Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--agents N` | 24 | Number of agents |
| `--names N` | 10 | Number of name tokens |
| `--rounds N` | 1000 | Rounds per run |
| `--runs N` | 40 | Number of runs |
| `--temperature T` | 0.9 | Softmax temperature |
| `--etas SPEC` | `3:1,2:1,1:1` | Eta ratios (η+:η-) |
| `--fracs SPEC` | `0.0,0.1,...,1.0` | Committed fractions (tipping) |
| `--outdir PATH` | `results/baselines` | Output directory |
| `--seed N` | 12345 | Random seed |
| `-v, --verbose` | 1 | Verbosity level |

### Demo Types
- `convergence` - Show convergence curves for different eta ratios
- `tipping` - Tipping point analysis with committed minorities
- `shock` - Shock response and recovery
- `eta_sweep` - Sweep eta parameter space

### Examples

```bash
# Dirichlet convergence
python -m src.scripts.run_baselines \
    --model dirichlet \
    --demo convergence \
    --runs 40

# Replicator-Logit convergence
python -m src.scripts.run_baselines \
    --model replicator \
    --demo convergence \
    --runs 40

# Both models comparison
python -m src.scripts.run_baselines \
    --model both \
    --demo convergence \
    --runs 20

# Tipping point analysis
python -m src.scripts.run_baselines \
    --model dirichlet \
    --demo tipping \
    --fracs 0.0,0.1,0.2,0.3,0.4,0.5 \
    --runs 50

# Shock response
python -m src.scripts.run_baselines \
    --model replicator \
    --demo shock \
    --rounds 2000 \
    --runs 30

# Custom eta ratios
python -m src.scripts.run_baselines \
    --model dirichlet \
    --demo convergence \
    --etas "8:1,4:1,2:1,1:1" \
    --runs 20

# High-resolution simulation
python -m src.scripts.run_baselines \
    --model both \
    --demo convergence \
    --agents 50 \
    --rounds 5000 \
    --runs 100

# Quick test
python -m src.scripts.run_baselines \
    --model dirichlet \
    --demo convergence \
    --rounds 500 \
    --runs 5 \
    -v -v
```

---

## 7. Figure Generation

Generate publication-ready figures from results.

```bash
python -m src.scripts.make_figures [OPTIONS]
```

### Required Arguments
| Argument | Description |
|----------|-------------|
| `--input PATH` | Input results directory |

### Optional Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--output PATH` | `paper_figures/` | Output directory |
| `--type TYPE` | `all` | Figure type (`all`, `ablation`, `beta`, `shock`, `baselines`, `hetero`, `prior`) |
| `--format FMT` | `png` | Output format (`png`, `pdf`, `svg`) |
| `--dpi N` | 300 | Resolution for raster formats |

### Examples

```bash
# Generate all figures
python -m src.scripts.make_figures \
    --input results/ \
    --output paper_figures/

# Only ablation figures
python -m src.scripts.make_figures \
    --input results/naming_game \
    --type ablation

# High-DPI PDF for publication
python -m src.scripts.make_figures \
    --input results/ \
    --output final_figures/ \
    --format pdf \
    --dpi 600

# Only baseline figures
python -m src.scripts.make_figures \
    --input results/baselines \
    --type baselines
```

---

## 8. Environment Variables

Set these before running commands:

```bash
# Required for OpenAI
export OPENAI_API_KEY="sk-..."

# Required for Gemini
export GOOGLE_API_KEY="..."

# Optional for gated HuggingFace models
export HF_TOKEN="hf_..."

# For testing without API calls
export MOCK_LLM=1

# For reproducible CUDA operations
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Or use a `.env` file:

```bash
# .env file
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
HF_TOKEN=hf_...
```

---

## 9. Common Patterns

### Quick Testing
Always start with small parameters to verify setup:

```bash
python -m src.scripts.run_naming_game \
    --preset gpt4o-mini \
    --agents 4 \
    --rounds 10 \
    --runs 1 \
    -v -v
```

### Reproducible Runs
Use `--seed` for reproducibility:

```bash
python -m src.scripts.run_naming_game \
    --preset qwen7b \
    --seed 42 \
    --runs 5
```

### Mock Mode (No API Calls)
Test without spending API credits:

```bash
MOCK_LLM=1 python -m src.scripts.run_naming_game \
    --preset gpt4o-mini \
    --runs 3
```

### Custom Output Directory
Organize results by date/experiment:

```bash
python -m src.scripts.run_naming_game \
    --preset gpt4o-mini \
    --outdir results/2026-01-31/ablation_gpt4o \
    --runs 10
```

### Verbose Output
Add `-v` for more logging, `-v -v` for debug:

```bash
python -m src.scripts.run_baselines \
    --model dirichlet \
    --demo convergence \
    -v -v
```

### Running Multiple Conditions in Parallel
Use background jobs or tmux:

```bash
# Background jobs
python -m src.scripts.run_naming_game --preset gpt4o-mini --conditions scored &
python -m src.scripts.run_naming_game --preset gpt4o-mini --conditions structure_only &
wait

# Or in tmux/screen sessions
tmux new-session -d -s exp1 'python -m src.scripts.run_naming_game --preset gpt4o-mini'
tmux new-session -d -s exp2 'python -m src.scripts.run_naming_game --preset qwen7b'
```

---

## Model Presets Reference

| Preset | Backend | Model ID | VRAM |
|--------|---------|----------|------|
| `gpt4o-mini` | OpenAI | gpt-4o-mini | API |
| `gpt4o` | OpenAI | gpt-4o | API |
| `gpt35-turbo` | OpenAI | gpt-3.5-turbo | API |
| `gemini-flash` | Gemini | gemini-1.5-flash | API |
| `gemini-pro` | Gemini | gemini-1.5-pro | API |
| `qwen7b` | HuggingFace | Qwen/Qwen2.5-7B-Instruct | ~8GB |
| `yi6b` | HuggingFace | 01-ai/Yi-1.5-6B-Chat | ~6GB |
| `tinyllama` | HuggingFace | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | ~2GB |
| `phi3_mini_4k_instruct` | HuggingFace | microsoft/Phi-3-mini-4k-instruct | ~4GB |

---

## Quick Reference Card

```bash
# Naming game ablations
python -m src.scripts.run_naming_game --preset MODEL --runs N

# Mixed models
python -m src.scripts.run_mixed_models --presets M1,M2 --fractions F1,F2

# Beta calibration
python -m src.scripts.run_beta_calibration --preset MODEL --temps T1,T2

# Prior probe
python -m src.scripts.run_prior_probe --preset MODEL --runs N

# Shock experiment
python -m src.scripts.run_shock --preset MODEL --shock-at 0.5

# Baselines
python -m src.scripts.run_baselines --model dirichlet --demo convergence

# Figures
python -m src.scripts.make_figures --input results/ --output figures/
```
