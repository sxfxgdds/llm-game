from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class AgentState:
    agent_id: str
    agent_type: str
    provider: str
    model_name: str
    temperature: float
    # 新增：智能体的社会声誉得分，初始化为 0.5 (中立)
    reputation: float = 0.5
    history: List[Dict[str, Any]] = field(default_factory=list)
    total_reward: float = 0.0
    
    def add_trial(self, trial_data: Dict[str, Any]):
        """记录每一轮的博弈结果并累加收益"""
        self.history.append(trial_data)
        self.total_reward += trial_data.get("reward", 0.0)

@dataclass
class GameState:
    round: int = 0
    agents: Dict[str, AgentState] = field(default_factory=dict)
    
    def get_agent(self, agent_id: str) -> AgentState:
        return self.agents[agent_id]