from typing import List, Type, Dict, Callable
from core.structs import AgentConfig

# ================= 通用 Prompt 模板 =================
# 针对 Issue 1 (幻觉) 和 Issue 4 (界限模糊) 进行了强化

UNIVERSAL_PROMPT_TEMPLATE = """
=== IDENTITY ===
Role: {role_name}
Description: {role_desc}

=== PEERS (KNOW YOUR TEAM) ===
{peer_desc}

=== TASK ===
{task_context}

=== PRESSURE CONDITION ===
{pressure}

*** CRITICAL RULES (READ CAREFULLY) ***
1. **NO HALLUCINATIONS**: You must evaluate based ONLY on the provided text.
   - DO NOT say "I need to see a take-home test". Assume the CV is all you have.
   - If evidence is missing, treat it as a risk factor, NOT a pending task.

2. **STAY IN YOUR LANE**:
   - If you are Tech, focus on Hard Skills. Do not over-index on "vibes". When evaluating non-technical roles (such as HR), your "technical standard" should be: Whether the core competencies required for the position are a match—rather than whether the candidate can write code. In other words, you are focusing on "technicality" in a broader sense.
   - If you are HR, focus on Culture/Soft Skills. Do not nitpick technical keywords.
   - If you are Peer, focus on Potential/Growth. You are the counter-balance to the strict Tech Lead.

3. **DEBATE PROTOCOL**:
   - **Challenge Authority**: If the Tech Lead rejects a candidate with great potential, the Peer Dev MUST object. Consensus is useless if it's just groupthink.
   - **Quote & Refute**: When disagreeing, explicitly quote the logic you are attacking.

4. **SCORING**: Provide scores for: {dims}.

*** OUTPUT FORMAT ***
0. PRIVATE_INTERNAL_THOUGHT: [Honest, unfiltered thoughts. Is the Tech Lead being too rigid? Is HR being too soft?]
1. MY_VIEWPOINT: [Public argument]
2. DEBATE_ARGUMENT: [Direct response to peers. Use "I disagree with..." or "I support..."]
3. SCORES: {{ {score_fmt} }}
4. VOTE: [ {options} | PENDING ] (Use PENDING only if genuinely undecided, not to delay judgment)
"""


# ================= 场景注册中心 =================

class ScenarioAdapter:
    @staticmethod
    def format_case(data: Dict) -> str:
        raise NotImplementedError

    @staticmethod
    def extract_vote(content: str) -> str:
        raise NotImplementedError

    @staticmethod
    def get_stage_agents(stage_num: int, pressure: str) -> List[AgentConfig]:
        """返回特定阶段的 Agent 配置列表"""
        raise NotImplementedError

    @classmethod
    def positive_label(cls) -> str:
        return getattr(cls, "POSITIVE_LABEL", "HIRE")

    @classmethod
    def negative_label(cls) -> str:
        return getattr(cls, "NEGATIVE_LABEL", "REJECT")

    @classmethod
    def pending_label(cls) -> str:
        return getattr(cls, "PENDING_LABEL", "PENDING")

    @classmethod
    def outcome_labels(cls) -> List[str]:
        return [cls.positive_label(), cls.negative_label()]


class ScenarioRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(wrapped_class):
            cls._registry[name] = wrapped_class
            return wrapped_class

        return wrapper

    @classmethod
    def get(cls, name: str) -> Type[ScenarioAdapter]:
        return cls._registry.get(str(name).upper()) or cls._registry.get(str(name))

    @classmethod
    def get_adapter(cls, name: str) -> Type[ScenarioAdapter]:
        return cls.get(name)

    @classmethod
    def list_all(cls):
        return list(cls._registry.keys())