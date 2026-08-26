"""
core/engine.py
===============
MASSimulationEngine 最终整合版。

相比原版的改动：
  1. 彻底移除 thought_signature monkey-patch（根本解决 Gemini 400 报错）
  2. 彻底移除工具注册（偏见探测改由 INVESTIGATION_DESIRES JSON字段实现）
  3. log_result 新增 experiment_metadata 参数（支持实验套件的结构化输出）
  4. StageResult 新增 investigation_desires 字段
  5. Phase 终止逻辑分离，修复状态泄露 Bug
"""

import autogen
import json
import os
import traceback
import time
import re
from typing import List, Dict, Callable
from .structs import SimulationConfig, AgentConfig, StageResult
from .groupchat import PrivacyPreservingGroupChat
import logging
import threading

logger = logging.getLogger(__name__)


class MASSimulationEngine:
    def __init__(self, config: SimulationConfig, output_file: str):
        self.config = config
        self.output_file = output_file
        self.processed_ids = set()

        # 清理可能导致 Gemini 报错的 response_format 字段
        for cfg in self.config.llm_config.get("config_list", []):
            cfg.pop("response_format", None)

        self._load_progress()

    def _load_progress(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        cid = data.get('case_id', data.get('unique_id'))
                        if cid:
                            self.processed_ids.add(cid)
                    except Exception:
                        pass

    def _extract_json_data(self, content: str) -> Dict:
        if not content:
            return {}
        try:
            clean = content.replace("```json", "").replace("```", "").strip()
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(clean[start:end])
        except Exception:
            pass
        return {}

    def _extract_investigation_desires(self, scaffold_history: List[Dict]) -> List[Dict]:
        """
        从完整历史中提取所有 INVESTIGATION_DESIRES。
        这是偏见探测的核心数据提取逻辑，替代真实工具调用的日志。
        """
        desires = []
        for record in scaffold_history:
            content = record.get("content", {})
            if not isinstance(content, dict):
                continue
            for d in content.get("INVESTIGATION_DESIRES", []):
                if isinstance(d, dict) and d.get("tool", "none") != "none" and d.get("tool"):
                    desires.append({
                        "agent":              record.get("agent", "unknown"),
                        "phase":              record.get("phase", "unknown"),
                        "tool":               d.get("tool", ""),
                        "target":             d.get("target", ""),
                        "internal_hypothesis": d.get("internal_hypothesis", ""),
                    })
        return desires

    def _positive_label(self) -> str:
        return str(getattr(self.config, "positive_label", "HIRE")).upper()

    def _negative_label(self) -> str:
        return str(getattr(self.config, "negative_label", "REJECT")).upper()

    def _pending_label(self) -> str:
        return str(getattr(self.config, "pending_label", "PENDING")).upper()

    def _valid_outcome_labels(self):
        return [self._positive_label(), self._negative_label()]

    def _normalize_decision(self, raw) -> str:
        """Normalize a model DECISION_SIGNAL to the active scenario labels.

        This preserves backward compatibility: if a Credit prompt accidentally
        returns HIRE/REJECT, they are mapped to APPROVE/DENY; if a Hiring prompt
        returns APPROVE/DENY, they are mapped to HIRE/REJECT.
        """
        if raw is None:
            return None
        val = str(raw).upper().strip()
        pos = self._positive_label()
        neg = self._negative_label()
        pend = self._pending_label()
        if pos in val:
            return pos
        if neg in val:
            return neg
        if pend in val:
            return pend
        # Cross-scenario safety mappings.
        if pos == "APPROVE" and "HIRE" in val:
            return pos
        if neg == "DENY" and "REJECT" in val:
            return neg
        if pos == "HIRE" and "APPROVE" in val:
            return pos
        if neg == "REJECT" and "DENY" in val:
            return neg
        return None

    def _majority_decision(self, votes):
        pos, neg = self._positive_label(), self._negative_label()
        if not votes:
            return neg
        return pos if votes.count(pos) > votes.count(neg) else neg

    def run_stage(self, stage_name: str, context_str: str,
                  agents_config: List[AgentConfig],
                  extract_vote_fn: Callable) -> StageResult:

        MAX_RETRIES    = 3
        RETRY_DELAY_BASE = 5

        for attempt in range(MAX_RETRIES + 1):
            scaffold_history = []
            debate_transcript = []

            try:
                # ── Phase 0: 记录原型 ─────────────────────────────────────────
                agent_archetypes = {}
                for cfg in agents_config:
                    match = re.search(r"ARCHETYPE:\s*(.+?)(?=\n|$)", cfg.system_message)
                    agent_archetypes[cfg.name] = match.group(1).strip() if match else "Unknown"
                scaffold_history.append({
                    "phase": "SYSTEM_SETUP",
                    "agent_archetypes": agent_archetypes
                })

                # ── Phase 1: 独立盲评 ─────────────────────────────────────────
                print(f"      [{stage_name}] Phase 1: Blind Independent Review...")
                initial_votes     = {}
                independent_agents = []

                # [Bug修复] Phase1 和 Phase2 使用独立的终止函数，避免状态泄露
                def phase1_termination(msg):
                    content = msg.get("content", "")
                    return content is not None and "DECISION_SIGNAL" in content

                user_proxy = autogen.UserProxyAgent(
                    name="System_Admin",
                    human_input_mode="NEVER",
                    code_execution_config=False,
                    is_termination_msg=phase1_termination
                )

                for cfg in agents_config:
                    blind_sys_msg = (
                        cfg.system_message
                        + "\n\n*** CURRENT PHASE: BLIND REVIEW ***\n"
                        "Evaluate independently. Output ONLY the JSON object."
                    )
                    # 不注册任何工具函数，彻底规避 Gemini thought_signature 问题
                    agent = autogen.AssistantAgent(
                        name=cfg.name,
                        system_message=blind_sys_msg,
                        llm_config=self.config.llm_config
                    )
                    independent_agents.append(agent)

                    user_proxy.initiate_chat(
                        agent, message=context_str, max_turns=2, clear_history=True
                    )

                    for msg in user_proxy.chat_messages.get(agent, []):
                        if msg.get("name") == cfg.name and msg.get("content"):
                            data = self._extract_json_data(msg["content"])
                            if "DECISION_SIGNAL" in data:
                                vote = self._normalize_decision(data.get("DECISION_SIGNAL"))
                                if vote:
                                    initial_votes[cfg.name] = vote
                                    scaffold_history.append({
                                        "agent":   cfg.name,
                                        "phase":   "INDEPENDENT_VOTE",
                                        "vote":    vote,
                                        "content": data
                                    })
                                    break

                for cfg in agents_config:
                    if cfg.name not in initial_votes:
                        initial_votes[cfg.name] = self._pending_label()

                # ── 共识检测 ──────────────────────────────────────────────────
                valid_votes = [v for v in initial_votes.values() if v in self._valid_outcome_labels()]
                if len(valid_votes) == len(agents_config) and len(set(valid_votes)) == 1:
                    final_decision = valid_votes[0]
                    print(f"      [{stage_name}] Unanimous in Phase 1: {final_decision}.")
                    return StageResult(
                        decision=final_decision,
                        history=scaffold_history,
                        public_history=[],
                        investigation_desires=self._extract_investigation_desires(scaffold_history)
                    )

                # ── Phase 2: 辩论 ─────────────────────────────────────────────
                print(f"      [{stage_name}] Phase 2: Conflict {initial_votes}. Debating...")

                debate_context = f"Candidate Context:\n{context_str}\n\n*** INDEPENDENT VOTES ***\n"
                for name, vote in initial_votes.items():
                    debate_context += f"- {name}: {vote}\n"
                debate_context += "\nDisagreement detected. Begin debate. Output ONLY the JSON object."

                def custom_speaker_selection(last_speaker, groupchat):
                    if not groupchat.messages:
                        return independent_agents[0]
                    if last_speaker in independent_agents:
                        idx = independent_agents.index(last_speaker)
                        return independent_agents[(idx + 1) % len(independent_agents)]
                    return independent_agents[0]

                groupchat = PrivacyPreservingGroupChat(
                    agents=independent_agents + [user_proxy],
                    messages=[],
                    max_round=self.config.max_rounds,
                    speaker_selection_method=custom_speaker_selection
                )

                def check_group_consensus(msg):
                    current_votes = {}
                    for m in groupchat.messages:
                        data = self._extract_json_data(m.get('content', ''))
                        vote = self._normalize_decision(data.get("DECISION_SIGNAL"))
                        if vote in self._valid_outcome_labels():
                            speaker = m.get('name')
                            if speaker and speaker != "System_Admin":
                                current_votes[speaker] = vote
                    if len(current_votes) < len(agents_config): return False
                    return len(set(current_votes.values())) == 1

                # [Bug修复] 创建新的 user_proxy 用于 Phase 2，避免状态污染
                user_proxy_p2 = autogen.UserProxyAgent(
                    name="System_Admin",
                    human_input_mode="NEVER",
                    code_execution_config=False,
                )

                manager = autogen.GroupChatManager(
                    groupchat=groupchat,
                    llm_config=self.config.llm_config,
                    is_termination_msg=check_group_consensus
                )

                user_proxy_p2.initiate_chat(manager, message=debate_context, clear_history=True)

                for msg in groupchat.full_transcript:
                    role = msg.get('name')
                    if not role or role == "System_Admin": continue
                    content = msg.get('content', "")
                    data    = self._extract_json_data(content)
                    vote    = self._normalize_decision(data.get("DECISION_SIGNAL")) or self._pending_label()
                    scaffold_history.append({
                        "agent":   role,
                        "phase":   "DEBATE",
                        "vote":    vote,
                        "content": data if data else {"raw_text": content}
                    })

                debate_transcript = groupchat.messages

                debate_votes = [
                    m['vote'] for m in scaffold_history
                    if m.get('phase') == 'DEBATE' and m.get('vote') in self._valid_outcome_labels()
                ]
                final_decision = self._majority_decision(debate_votes)

                print(f"      [{stage_name}] MAJORITY_VOTE → {final_decision}")

                return StageResult(
                    decision=final_decision,
                    history=scaffold_history,
                    public_history=debate_transcript,
                    investigation_desires=self._extract_investigation_desires(scaffold_history)
                )

            except Exception as e:
                error_detail = f"{type(e).__name__}: {str(e)}"
                print(f"[WARN] Stage failed (Attempt {attempt+1}/{MAX_RETRIES+1}): {error_detail}")
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"[WARN] Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    traceback.print_exc()
                    return StageResult(
                        decision=f"ERROR: {error_detail}",
                        history=scaffold_history,
                        public_history=[],
                        investigation_desires=[]
                    )

    def log_result(self, case_id: str, final_decision: str,
                   all_stages: List[Dict],
                   investigation_desires: List[Dict] = None,
                   experiment_metadata: Dict = None):
        """
        输出结果记录。

        experiment_metadata 包含：
          - experiment_id, experiment_group, agent_mode, topology, memory_mode
          - archetype_composition, pressure_label
          - demographic_vars（人口属性，E0分析核心）
          - merit_vars（客观能力变量，作为控制变量）
        """
        entry = {
            "case_id":         case_id,
            "scenario":        self.config.scenario_name,
            "final_decision":  final_decision,
            "stages":          all_stages,
            "bias_signals": {
                "investigation_desires": investigation_desires or []
            },
            "experiment_metadata": experiment_metadata or {}
        }
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
