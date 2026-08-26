"""
core/structs.py
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any


class TopologyType(Enum):
    DEMOCRATIC = "democratic"
    PIPELINE   = "pipeline"


@dataclass
class AgentConfig:
    name: str
    system_message: str


@dataclass
class SimulationConfig:
    scenario_name:        str
    topology:             TopologyType
    max_rounds:           int  = 20
    llm_config:           Dict = field(default_factory=dict)
    pressure_instruction: str  = ""

    # Scenario-level decision labels. Hiring keeps HIRE/REJECT;
    # Credit uses APPROVE/DENY. The engine treats the first label as
    # the stage-gating positive decision and the second as the negative decision.
    positive_label:       str  = "HIRE"
    negative_label:       str  = "REJECT"
    pending_label:        str  = "PENDING"


@dataclass
class StageResult:
    decision:             str
    history:              List[Dict]   # 完整历史（含 PRIVATE_INTERNAL_THOUGHT）
    public_history:       List[Dict]   # 公开历史（用于 Stage2 上下文）
    investigation_desires: List[Dict]  = field(default_factory=list)
    # investigation_desires 字段说明：
    # 每条记录：{agent, phase, tool, target, internal_hypothesis}
    # 这是偏见探测的代理指标——Agent为哪类候选人主动发起额外调查？
