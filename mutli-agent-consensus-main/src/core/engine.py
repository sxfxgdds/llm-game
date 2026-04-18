"""
Shared game engine for population naming games (Modified for Prisoner's Dilemma).

This module provides the core simulation loop, heavily modified to test
Reputation-based Adaptive Exploration in the Repeated Prisoner's Dilemma.
"""

import time
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable

from .state import AgentState, GameState, RunResult
from .pairing import pair_indices, chunk_pairs, shared_allowed_for_round
# 引入你在 prompts.py 中新定义的囚徒困境 Prompt 生成器
from .prompts import PromptBuilder, PromptConfig, get_pd_reputation_prompt
from .parsing import extract_allowed_choice
from .metrics import rolling_mean, aggregate_curves
from .io import (
    TrialsWriter, TrialRow, hash_prompt,
    save_raw_response, init_run_dir, dump_resolved_config
)
from .utils import set_global_seed, make_simple_tokens, make_nonce_tokens

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the game engine."""

    # Game parameters
    n_agents: int = 24
    n_rounds: int = 200
    n_names: int = 10  # 在 PD 游戏中此参数可忽略，我们强制为2 ("C", "D")
    history_length: int = 6
    temperature: float = 1.0  # 基础温度，将在循环中被声誉机制覆盖

    # Prompt configuration
    prompt_variant: str = "scored"
    use_nonce_tokens: bool = False

    # Shock parameters (optional)
    shock_round: Optional[int] = None
    flip_goal: bool = False

    # Logging
    store_raw_responses: bool = False
    verbosity: int = 1

    # Run metadata
    run_id: str = ""
    experiment: str = "naming_game"
    variant_id: str = ""
    seed: int = 12345
    repeat_idx: int = 0


@dataclass
class AgentContext:
    """Runtime context for an agent."""
    agent_id: int
    agent_type: str  # preset name
    client: Any  # LLM client
    history: deque = field(default_factory=lambda: deque(maxlen=6))
    cum_reward: float = 0.0
    # 【新增】智能体的社会声誉，初始化为 0.5 (中立)
    reputation: float = 0.5

    def __post_init__(self):
        if not hasattr(self, 'history') or self.history is None:
            self.history = deque(maxlen=6)


def calculate_pd_payoff(choice_a: str, choice_b: str):
    """
    【新增】计算囚徒困境的收益矩阵。
    T(Temptation)=5, R(Reward)=3, P(Punishment)=1, S(Sucker)=0
    """
    if choice_a == "C" and choice_b == "C": return 3.0, 3.0
    if choice_a == "C" and choice_b == "D": return 0.0, 5.0
    if choice_a == "D" and choice_b == "C": return 5.0, 0.0
    if choice_a == "D" and choice_b == "D": return 1.0, 1.0
    return 0.0, 0.0 # 处理无效或异常输出


def run_population_game(
    agent_types: List[str],
    clients: Dict[str, Any],
    config: EngineConfig,
    trials_writer: Optional[TrialsWriter] = None,
    run_path: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> RunResult:
    """
    Run a single population game (Modified for PD & Reputation).
    """
    if rng is None:
        rng = random.Random(config.seed)

    N = config.n_agents
    H = config.history_length

    # 【修改】强制动作空间为合作 (C) 与背叛 (D)
    allowed_round = ["C", "D"]
    W = 2

    # 初始化智能体
    agents = []
    for i, agent_type in enumerate(agent_types):
        agents.append(AgentContext(
            agent_id=i,
            agent_type=agent_type,
            client=clients[agent_type],
            history=deque(maxlen=H),
            cum_reward=0.0,
            reputation=0.5  # 初始声誉
        ))

    # Tracking
    per_round_success = []  # 在 PD 中，我们可以将其视为 "合作率 (Cooperation Rate)"
    per_round_choices = []
    invalids = 0
    total_retries = 0
    total_trials = 0

    within_type_matches = []
    cross_type_matches = []

    in_shock_phase = False
    shock_round = config.shock_round

    # Main game loop
    for round_idx in range(config.n_rounds):
        phase = "na"
        if shock_round is not None:
            if round_idx < shock_round:
                phase = "pre_shock"
            else:
                phase = "post_shock"
                if not in_shock_phase:
                    in_shock_phase = True
                    if config.verbosity >= 1:
                        logger.info(f"  Shock applied at round {round_idx}")

        # Generate pairs
        pairs = pair_indices(N, rng)

        round_cooperations = 0 # 记录本轮总共有多少次合作
        round_choices = []

        for i, j in pairs:
            agent_i = agents[i]
            agent_j = agents[j]

            # 【核心机制 1：自适应探索 (Adaptive Exploration)】
            # 基于声誉计算动态温度：声誉越高，温度越低；声誉越低，温度越高
            temp_i = max(0.1, 1.1 - agent_i.reputation)
            temp_j = max(0.1, 1.1 - agent_j.reputation)

            # 【核心机制 2：生成声誉感知的 Prompt】
            prompt_i = get_pd_reputation_prompt(
                agent_reputation=agent_i.reputation,
                partner_reputation=agent_j.reputation,
                history=list(agent_i.history),
                history_length=H
            )
            prompt_j = get_pd_reputation_prompt(
                agent_reputation=agent_j.reputation,
                partner_reputation=agent_i.reputation,
                history=list(agent_j.history),
                history_length=H
            )

            # Query agents (传入动态温度)
            t0 = time.time()
            result_i = agent_i.client.generate_choice(
                prompt_i,
                allowed_round,
                temperature=temp_i,
                seed=config.seed + round_idx * 1000 + i,
            )
            latency_i = (time.time() - t0) * 1000

            t0 = time.time()
            result_j = agent_j.client.generate_choice(
                prompt_j,
                allowed_round,
                temperature=temp_j,
                seed=config.seed + round_idx * 1000 + j,
            )
            latency_j = (time.time() - t0) * 1000

            # Extract choices
            choice_i = result_i.choice if result_i.valid else rng.choice(allowed_round)
            choice_j = result_j.choice if result_j.valid else rng.choice(allowed_round)

            if not result_i.valid: invalids += 1
            if not result_j.valid: invalids += 1
            total_retries += result_i.retries + result_j.retries

            # 【核心机制 3：计算囚徒困境收益】
            reward_i, reward_j = calculate_pd_payoff(choice_i, choice_j)

            agent_i.cum_reward += reward_i
            agent_j.cum_reward += reward_j

            # 记录历史：这里记录的是 (对手选择, 我的收益)
            agent_i.history.append({"partner_choice": choice_j, "reward": reward_i})
            agent_j.history.append({"partner_choice": choice_i, "reward": reward_j})

            # 【核心机制 4：非对称状态依赖的声誉更新】
            alpha = 0.2
            target_i = 1.0 if choice_i == "C" else 0.0
            target_j = 1.0 if choice_j == "C" else 0.0
            agent_i.reputation += alpha * (target_i - agent_i.reputation)
            agent_j.reputation += alpha * (target_j - agent_j.reputation)

            # 统计合作率
            if choice_i == "C": round_cooperations += 1
            if choice_j == "C": round_cooperations += 1
            round_choices.extend([choice_i, choice_j])
            total_trials += 2

            # Verbose logging
            if config.verbosity >= 2:
                logger.info(
                    f"      [{i:2d}↔{j:2d}] "
                    f"A({agent_i.reputation:.2f}): {choice_i} vs B({agent_j.reputation:.2f}): {choice_j} | "
                    f"Rewards: {reward_i}, {reward_j}"
                )

            # Log to trials CSV (在 reward 列复用收益，同时我们把声誉信息写进日志)
            if trials_writer:
                trials_writer.log({
                    "run_id": config.run_id,
                    "experiment": config.experiment,
                    "variant_id": config.variant_id,
                    "seed": config.seed,
                    "repeat_idx": config.repeat_idx,
                    "round": round_idx,
                    "phase": phase,
                    "agent_id": i,
                    "agent_type": agent_i.agent_type,
                    "partner_id": j,
                    "partner_type": agent_j.agent_type,
                    "provider": result_i.meta.get("provider", ""),
                    "model_name": result_i.meta.get("model", ""),
                    "temperature": temp_i, # 记录真实使用的动态温度
                    "H": H,
                    "N": N,
                    "W": W,
                    "prompt_hash": hash_prompt(prompt_i),
                    "allowed_set_id": "C_D_PD",
                    "allowed_order_id": "C_D_PD",
                    "prompt_variant": config.prompt_variant,
                    "choice": choice_i,
                    "choice_valid": result_i.valid,
                    "partner_choice": choice_j,
                    "match": (choice_i == "C"), # 复用 match 列存储该 Agent 是否合作 (方便旧代码画图)
                    "reward": reward_i,
                    "cum_reward": agent_i.cum_reward,
                    "latency_ms": latency_i,
                    "retries": result_i.retries,
                    "raw_response_path": f"Reputation: {agent_i.reputation:.3f}", # 复用未使用的列记录声誉
                })

                trials_writer.log({
                    "run_id": config.run_id,
                    "experiment": config.experiment,
                    "variant_id": config.variant_id,
                    "seed": config.seed,
                    "repeat_idx": config.repeat_idx,
                    "round": round_idx,
                    "phase": phase,
                    "agent_id": j,
                    "agent_type": agent_j.agent_type,
                    "partner_id": i,
                    "partner_type": agent_i.agent_type,
                    "provider": result_j.meta.get("provider", ""),
                    "model_name": result_j.meta.get("model", ""),
                    "temperature": temp_j,
                    "H": H,
                    "N": N,
                    "W": W,
                    "prompt_hash": hash_prompt(prompt_j),
                    "allowed_set_id": "C_D_PD",
                    "allowed_order_id": "C_D_PD",
                    "prompt_variant": config.prompt_variant,
                    "choice": choice_j,
                    "choice_valid": result_j.valid,
                    "partner_choice": choice_i,
                    "match": (choice_j == "C"),
                    "reward": reward_j,
                    "cum_reward": agent_j.cum_reward,
                    "latency_ms": latency_j,
                    "retries": result_j.retries,
                    "raw_response_path": f"Reputation: {agent_j.reputation:.3f}",
                })

        # 记录本轮群体的整体合作率 (Cooperation Rate) 代替原本的 Match Rate
        match_rate = round_cooperations / (n_pairs * 2) if n_pairs > 0 else 0.0
        per_round_success.append(match_rate)
        per_round_choices.append(round_choices)

        if config.verbosity >= 1:
            log_interval = max(1, min(10, config.n_rounds // 10))
            if (round_idx + 1) % log_interval == 0 or round_idx == config.n_rounds - 1:
                avg_so_far = sum(per_round_success) / len(per_round_success)
                recent_avg = sum(per_round_success[-log_interval:]) / min(log_interval, len(per_round_success))
                pct = (round_idx + 1) / config.n_rounds * 100
                logger.info(
                    f"    Round {round_idx + 1:4d}/{config.n_rounds} ({pct:5.1f}%) | "
                    f"Coop Rate={match_rate:.2f} recent={recent_avg:.2f} avg={avg_so_far:.2f}"
                )

    invalid_rate = 100 * invalids / total_trials if total_trials > 0 else 0.0
    within_rate = sum(within_type_matches) / len(within_type_matches) if within_type_matches else None
    cross_rate = sum(cross_type_matches) / len(cross_type_matches) if cross_type_matches else None

    return RunResult(
        per_round_success=per_round_success,
        per_round_choices=per_round_choices,
        invalid_rate=invalid_rate,
        total_retries=total_retries,
        within_type_rate=within_rate,
        cross_type_rate=cross_rate,
        config=config.__dict__,
    )


def compute_run_summary(
    results: List[RunResult],
    config: EngineConfig,
    config_digest: str = "",
) -> Dict[str, Any]:
    """
    Compute summary statistics from multiple run results.
    """
    import numpy as np

    all_success = [r.per_round_success for r in results]
    final_rates = [s[-1] if s else 0 for s in all_success]

    def find_consensus(success_curve, threshold=0.9, streak=10):
        count = 0
        for i, s in enumerate(success_curve):
            if s >= threshold:
                count += 1
                if count >= streak:
                    return i - streak + 1
            else:
                count = 0
        return None

    consensus_times = [find_consensus(s) for s in all_success]
    valid_times = [t for t in consensus_times if t is not None]

    from .metrics import compute_entropy, compute_concentration
    final_choices = []
    for r in results:
        if r.per_round_choices:
            final_choices.extend(r.per_round_choices[-1])

    entropy = compute_entropy(final_choices) if final_choices else None
    dominant_share = compute_concentration(final_choices) if final_choices else None

    n_invalid = sum(r.invalid_rate * len(r.per_round_success) * config.n_agents / 100 for r in results)
    n_retries = sum(r.total_retries for r in results)
    n_trials = sum(len(r.per_round_success) * config.n_agents for r in results)

    within_rates = [r.within_type_rate for r in results if r.within_type_rate is not None]
    cross_rates = [r.cross_type_rate for r in results if r.cross_type_rate is not None]

    return {
        "run_id": config.run_id,
        "experiment": config.experiment,
        "variant_id": config.variant_id,
        "config_digest": config_digest,
        "n_trials_total": int(n_trials),
        "n_invalid": int(n_invalid),
        "n_retries_total": int(n_retries),
        "final_match_rate": float(np.mean(final_rates)), # 实际上现在代表 final_coop_rate
        "final_match_rate_std": float(np.std(final_rates)),
        "time_to_consensus": float(np.mean(valid_times)) if valid_times else None,
        "dominant_share_final": dominant_share,
        "entropy_final": entropy,
        "within_type_match_final": float(np.mean(within_rates)) if within_rates else None,
        "cross_type_match_final": float(np.mean(cross_rates)) if cross_rates else None,
        "notes": {
            "n_runs": len(results),
            "n_agents": config.n_agents,
            "n_rounds": config.n_rounds,
        }
    }