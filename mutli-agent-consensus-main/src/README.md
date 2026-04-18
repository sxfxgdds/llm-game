# LLM Social Conventions — Clean Codebase

Multi-agent naming-game experiments with LLMs and mechanistic baselines.

## Overview

This codebase runs multi-agent naming-game experiments with:
- **Closed LLMs**: OpenAI GPT, Google Gemini
- **Open-source LLMs**: HuggingFace Transformers (4-bit quantized)
- **Mechanistic baselines**: Dirichlet/Pólya, Replicator–Logit

It produces standardized trial logs and figure-ready aggregates.

## Key Concepts

- **Naming game core loop**: Random pairing, prompt with allowed labels + rolling history, one-token constrained response, match reward
- **β-calibration**: Estimate score→logit sensitivity per temperature; test τβ invariance
- **Heterogeneity**: Assign each agent a model type; analyze within-type vs cross-type coordination
- **Prior probes**: Full-string T=0 scoring to identify label priors

## Directory Structure

```
src/
├── core/           # Pairing, prompts, parsing, state, metrics, I/O, engine
│   ├── io.py       # Standardized output (trials.csv, run_summary.json)
│   ├── engine.py   # Shared game engine for all experiments
│   └── ...
├── llms/           # Provider adapters (separate files per provider)
│   ├── base.py     # ChoiceResult, CandidateScore, BaseLLMClient
│   ├── openai_client.py
│   ├── gemini_client.py
│   ├── huggingface_client.py
│   └── presets.py
├── baselines/      # Dirichlet and Replicator–Logit simulations
├── experiments/    # Runnable experiment definitions
│   ├── naming_game.py
│   ├── mixed_models.py    # Heterogeneous cohorts (NEW)
│   ├── beta_calibration.py
│   ├── prior_probe.py
│   └── shock.py
├── analysis/       # Aggregation + plots + paper tables
├── diagnostics/    # Reviewer-facing diagnostic tools (NEW)
│   ├── tokenizer_parity.py   # Build equal-tokenization label sets
│   ├── first_piece_vs_full.py # Compare scoring methods
│   └── prior_gap.py          # Analyze model priors
├── config/         # YAML experiment configurations
├── scripts/        # CLI entrypoints
├── tests/          # Unit tests (NEW)
└── requirements.txt
```

## Installation

```bash
# Core dependencies
pip install -r src/requirements.txt

# For OpenAI experiments
echo "OPENAI_API_KEY=your-key-here" > .env

# For HuggingFace experiments (optional)
pip install torch transformers accelerate bitsandbytes

# For llama.cpp experiments (optional)
pip install llama-cpp-python
```

## Quick Start

### 1. Naming Game Ablations (OpenAI)

```bash
python -m src.scripts.run_naming_game \
    --preset gpt4o-mini \
    --agents 24 \
    --rounds 200 \
    --runs 10 \
    --conditions scored structure_only no_score_in_history
```

### 2. Naming Game (Open-Source LLM)

```bash
python -m src.scripts.run_naming_game \
    --preset qwen7b \
    --agents 10 \
    --rounds 50 \
    --runs 3
```

### 3. Beta Calibration

```bash
python -m src.scripts.run_beta_calibration \
    --preset gpt4o-mini \
    --temps 0.5,1.0,1.5,2.0 \
    --balanced
```

### 4. Prior Probe

```bash
python -m src.scripts.run_prior_probe \
    --preset qwen7b \
    --agents 10 \
    --runs 5
```

### 5. Shock Experiments

```bash
python -m src.scripts.run_shock \
    --preset gpt4o-mini \
    --shock-at 0.5 \
    --flip-goal \
    --runs 10
```

### 6. Mixed-Model Experiments (Heterogeneous Cohorts)

```bash
python -m src.scripts.run_mixed_models \
    --presets gpt4o-mini,gemini-flash \
    --fractions 0.5,0.5 \
    --agents 24 \
    --rounds 200 \
    --runs 10
```

### 7. Baseline Simulations

```bash
python -m src.scripts.run_baselines \
    --model dirichlet \
    --demo convergence \
    --runs 40
```

### 8. Generate Figures

```bash
python -m src.scripts.make_figures \
    --input results/ \
    --output paper_figures/
```

## Configuration Files

YAML configs in `src/config/`:

- `naming_game.yaml` - Ablation experiments
- `mixed_models.yaml` - Heterogeneous cohort experiments
- `beta_calibration.yaml` - Score sensitivity testing
- `shock.yaml` - Reward-flip experiments
- `prior_probe.yaml` - Prior alignment testing
- `baselines.yaml` - Mechanistic model simulations

### Config Schema (B4)

Every config must include:

```yaml
run:
  out_dir: results/experiment
  run_name: null  # auto-generate
  seeds: [0, 1, 2, 3, 4]
  repeats_per_seed: 1

models:
  presets: [gpt4o-mini]

game:
  W: 10          # number of labels
  N: 24          # number of agents
  H: 6           # history length
  rounds: 200
  temperature: 1.0
  prompt_variant: scored

logging:
  store_prompts: false
  store_raw_responses: false
```

## Model Presets

| Preset | Backend | Model |
|--------|---------|-------|
| `gpt4o-mini` | OpenAI | gpt-4o-mini |
| `gpt4o` | OpenAI | gpt-4o |
| `qwen7b` | HuggingFace | Qwen/Qwen2.5-7B-Instruct |
| `yi6b` | HuggingFace | 01-ai/Yi-1.5-6B-Chat |
| `tinyllama` | HuggingFace | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| `gemini-flash` | Gemini | gemini-1.5-flash |

## Output Layout (Standardized)

```
results/{experiment}/{run_id}/
├── config.yaml           # Resolved configuration
├── trials.csv            # Trial-level data (standardized schema)
├── run_summary.json      # Run summary statistics
├── figures/*.png         # Visualizations
└── raw/                  # Raw API responses (if enabled)
```

### Standard Trial Schema (trials.csv)

Every row includes (30 columns):
- Identifiers: `run_id`, `experiment`, `variant_id`, `seed`, `repeat_idx`
- Game state: `round`, `phase`, `agent_id`, `agent_type`, `partner_id`, `partner_type`
- Model: `provider`, `model_name`, `temperature`
- Parameters: `H`, `N`, `W`, `prompt_variant`
- Results: `choice`, `choice_valid`, `partner_choice`, `match`, `reward`, `cum_reward`
- Performance: `latency_ms`, `retries`

### Standard Summary Schema (run_summary.json)

- Identifiers: `run_id`, `experiment`, `variant_id`, `config_digest`
- Counts: `n_trials_total`, `n_invalid`, `n_retries_total`
- Metrics: `final_match_rate`, `time_to_consensus`, `dominant_share_final`, `entropy_final`
- Heterogeneity: `within_type_match_final`, `cross_type_match_final`

## Python API

### Basic Usage

```python
from src.llms.presets import create_client
from src.experiments.naming_game import NamingGameExperiment, NamingGameConfig

# Create LLM client with seed for reproducibility
client = create_client(preset="gpt4o-mini", temperature=0.3, seed=42)

# Configure experiment
config = NamingGameConfig(
    n_agents=24,
    n_rounds=200,
    n_runs=10,
    conditions=["scored", "structure_only"],
)

# Run experiment
experiment = NamingGameExperiment(client, config)
results = experiment.run_all()
```

### Mixed-Model (Heterogeneous) Experiments

```python
from src.experiments import MixedModelsConfig, CohortComposition, MixedModelsExperiment

config = MixedModelsConfig(
    composition=[
        CohortComposition(preset="gpt4o-mini", fraction=0.5),
        CohortComposition(preset="gemini-flash", fraction=0.5),
    ],
    n_agents=24,
    n_rounds=200,
    seeds=[0, 1, 2, 3, 4],
)

experiment = MixedModelsExperiment(config)
summary = experiment.run()

print(f"Within-type match: {summary['within_type_match_final']:.3f}")
print(f"Cross-type match: {summary['cross_type_match_final']:.3f}")
```

### LLM Client Interface

```python
from src.llms import OpenAIClient, ChoiceResult

client = OpenAIClient(model="gpt-4o-mini", seed=42)

# New generate_choice method with retry logic
result: ChoiceResult = client.generate_choice(
    prompt="Pick one: w0, w1, w2",
    allowed_labels=["w0", "w1", "w2"],
    temperature=1.0,
)

print(f"Choice: {result.choice}, Valid: {result.valid}, Retries: {result.retries}")
print(f"Metadata: {result.meta}")
```

### Diagnostics

```python
from src.diagnostics import (
    build_parity_labelset,
    analyze_labelset,
    compute_prior_gap,
)

# Build tokenizer-parity labels for fair heterogeneous experiments
labels = build_parity_labelset(
    base_vocab=["alpha", "beta", "gamma", ...],
    tokenizers={"gpt4": gpt4_tok, "qwen": qwen_tok},
    target="single_token",
    k=10,
)

# Analyze prior preferences
gap_result = compute_prior_gap(client, prompt, labels)
print(f"Top label: {gap_result.top_label}, Gap: {gap_result.gap:.2f}")
```

## Ablation Conditions

| Condition | History | Outcome Words | Numeric Scores | Goal Line |
|-----------|---------|---------------|----------------|-----------|
| `scored` | ✓ | ✓ | ✓ | ✓ |
| `structure_only` | ✓ | ✓ | ✗ | ✓ |
| `no_score_in_history` | ✓ | ✗ | ✗ | ✓ |
| `no_score_no_goal` | ✓ | ✗ | ✗ | ✗ |

## Reproducibility

Seeds are supported at multiple levels:

| Component | Reproducibility | Notes |
|-----------|----------------|-------|
| Baselines | ✅ Full | NumPy/Python seeds |
| HuggingFace | ✅ Full | PyTorch generator + `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| GGUF | ✅ Full | llama.cpp internal seeding |
| OpenAI | ⚠️ Best-effort | API seed param (check `system_fingerprint`) |
| Gemini | ❌ None | No API support |

```bash
# For fully reproducible HuggingFace runs:
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m src.scripts.run_naming_game --preset qwen7b --seed 42
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `HF_TOKEN` | HuggingFace token (for gated models) |
| `MOCK_LLM` | Set to "1" for testing without API calls |
| `CUBLAS_WORKSPACE_CONFIG` | Set to `:4096:8` for CUDA reproducibility |

## GPU Memory Requirements

| Model | Approximate VRAM |
|-------|------------------|
| TinyLlama (1.1B) | ~2 GB |
| Yi 1.5 (6B) | ~6 GB |
| Qwen 2.5 (7B) | ~8 GB |

All HuggingFace models use 4-bit quantization by default.

## Tests

```bash
# Install pytest
pip install pytest

# Run all tests
python -m pytest src/tests/ -v

# Run specific test file
python -m pytest src/tests/test_parsing.py -v
```

## Citation

```bibtex
@article{llm_social_conventions,
  title={Emergent Social Conventions in LLM Populations},
  author={...},
  year={2026}
}
```