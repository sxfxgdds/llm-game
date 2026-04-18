# From Prompts to Conventions: A Study of Multi-Agent LLM Consensus

Codebase for multi-agent LLM consensus experiments (naming game, shocks, tipping, prior probes, beta calibration, gap-deviation analysis, and mechanistic baselines).

## Repository Purpose

This repository contains the implementation used for the paper above, with experiment runners in `src/scripts`, experiment logic in `src/experiments`, model clients in `src/llms`, and figure/data utilities in `src/plotting`.

## Quick Start

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create your local environment file from the template:

```bash
cp .env.template .env
```

3. Edit `.env` and fill in the keys you need for your chosen backend(s).

4. Run from the repository root (recommended):

```bash
python -m src.scripts.run_naming_game --preset gpt4o-mini --runs 1 --rounds 20
```

## Environment Variables

Use `.env.template` as the source of truth. Copy it to `.env` and set values there.

- `OPENAI_API_KEY`: required for OpenAI presets
- `GROQ_API_KEY`: required for Groq presets
- `GOOGLE_API_KEY`: required for Gemini presets
- `HUGGINGFACEHUB_API_TOKEN`: supported for HF hosted inference
- `HF_TOKEN`: also supported by code paths (recommended to set this too for compatibility)
- `MOCK_LLM=1`: optional, runs without external API calls for quick smoke tests

## Important Path Notes (Public Users)

Update paths to match your machine and filesystem layout.

- Most runners default to writing under `results/...`; override with `--outdir`.
- Any command examples using local files should be adapted to your directory structure.
- `src/plotting/plot_naming_game_os.py` contains explicit path placeholders (`PATH_TO_..._TRIALS_CSV`) and must be edited or overridden via CLI args before use.
- If you use custom label files for OS runs, pass your own file path via `--allowed-path`.

## Backend and Model Modes

The code supports several execution modes. Use the runner that matches your setup.

### 1) Local open-source models on your machine (HF Transformers)

- Backend: `hf`
- No hosted inference required; model weights run locally
- Typical use:

```bash
python -m src.scripts.run_naming_game_os --preset hfapi-arch-router --runs 3 --rounds 200
```

### 2) Hugging Face hosted inference API

- Backend: `hf_api`
- Requires HF token in environment
- Typical use:

```bash
python -m src.scripts.run_naming_game_os_hf --preset hfapi-arch-router --temperature 1 --agents 10 --history 10 --runs 3 --rounds 200
```

### 3) Groq API models

- Backend: `groq`
- Requires `GROQ_API_KEY`
- Typical use:

```bash
python -m src.scripts.run_naming_game_os_groq --preset groq-llama-3.1-8b-instant --temperature 1 --agents 10 --history 10 --runs 3 --rounds 100
```

### 4) Mixed OS backends in one run (HF local + HF API + Groq)

- Backends: `hf`, `hf_api`, `groq` together
- Typical use:

```bash
python -m src.scripts.run_naming_game_all_os --presets groq-llama-3.1-8b-instant hfapi-arch-router groq-llama-4-scout-17b-16e-instruct --temperature 1 --agents 10 --history 10 --runs 3 --rounds 200
```

### 5) Closed API models (OpenAI, Gemini)

- OpenAI/Gemini are supported by the general runners:

```bash
python -m src.scripts.run_naming_game --preset gpt4o-mini --agents 24 --runs 3 --rounds 200
python -m src.scripts.run_naming_game --preset gemini-2-flash --agents 24 --runs 3 --rounds 200
```

### 6) Optional GGUF local backend

- Use explicit backend/model path:

```bash
python -m src.scripts.run_naming_game --backend gguf --model path/to/model.gguf --runs 3 --rounds 200
```

## Main Experiment Runners

- Naming game: `src/scripts/run_naming_game.py`
- OS naming game (local HF): `src/scripts/run_naming_game_os.py`
- OS naming game (Groq-capable): `src/scripts/run_naming_game_os_groq.py`
- OS naming game (HF API): `src/scripts/run_naming_game_os_hf.py`
- OS naming game (all OS backends): `src/scripts/run_naming_game_all_os.py`
- Prior probe: `src/scripts/run_prior_probe.py`
- Prior probe (Groq): `src/scripts/run_prior_probe_groq.py`
- Prior probe (all OS backends): `src/scripts/run_prior_probe_all_os.py`
- Beta calibration: `src/scripts/run_beta_calibration.py`
- Gap-deviation (Corollary-1 evidence): `src/scripts/run_gap_deviation.py`
- Shock experiments: `src/scripts/run_shock.py`
- Tipping experiments: `src/scripts/run_tipping.py`
- Mixed-model populations: `src/scripts/run_mixed_models.py`
- Mechanistic baselines: `src/scripts/run_baselines.py`
- Sweep runners: `src/scripts/run_naming_game_sweep.py`, `src/scripts/run_naming_game_sweep_parallel.py`

## Plotting and Aggregation

Utilities live in `src/plotting`.

Typical flow for tipping or shock:

```bash
python -m src.plotting.load_tipping_trials --input results/tipping --output results/tipping/tipping_all_trials.csv
python -m src.plotting.plot_tipping --input results/tipping/tipping_all_trials.csv --outdir results/tipping/figures
```

```bash
python -m src.plotting.load_shock_trials --input results/shock --output results/shock/shock_all_trials.csv
python -m src.plotting.plot_shock --input results/shock/shock_all_trials.csv --outdir results/shock/figures
```

Gap-deviation experiment flow:

```bash
python -m src.scripts.run_gap_deviation --presets gpt4o-mini gpt41-mini --n-label-sets 8 --W 10 --reference-tau 1.0 --gap-trials 300 --taus 0.2,0.5,0.7,1.0,1.5,2.0 --samples-per-tau 400 --top-logprobs 20 --seed 12345
python -m src.plotting.plot_gap_deviation --run-dir results/gap_deviation
```

## Presets and Configuration

- Presets are defined in `src/llms/presets.py`.
- Config templates are in `src/config/*.yaml`.
- You can run by preset (`--preset ...`) or explicit backend/model (`--backend ... --model ...`).

## Reproducibility Notes

- Seeds are passed throughout runners.
- API providers may not guarantee strict determinism.
- Local HF/GGUF runs are generally the best option for tighter reproducibility.

## Project Layout

```text
src/
  baselines/     # Dirichlet and replicator-logit baselines
  config/        # YAML templates
  core/          # Engine, parsing, prompts, metrics, I/O
  diagnostics/   # Tokenization and prior diagnostics
  experiments/   # Experiment implementations
  llms/          # Client backends + preset registry
  plotting/      # Loaders and plotting scripts
  scripts/       # CLI entry points
  tests/         # Unit tests
```

## Citation

```bibtex
@article{paper_placeholder_2026,
  title={From Prompts to Conventions: A Study of Multi-Agent LLM Consensus},
  author={...},
  year={2026}
}
```
