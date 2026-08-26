"""
run_suite.py
============
实验套件统一入口。支持两级断点续跑：
  - 套件级：(experiment_id, batch_file) 粒度，已完成的批次直接跳过
  - Case级：已写入 .jsonl 的 case_id，重启后由 engine._load_progress() 恢复

中断后重启只需运行与上次完全相同的命令，进度自动恢复。

用法：
    python run_suite.py --data_file data/Hiring/Java_Developer --suite full --auto_analyze
    python run_suite.py --data_file data/Hiring/Java_Developer --suite E5
    python run_suite.py --data_file data/Hiring/Java_Developer --suite minimal --auto_analyze

Agent模式说明：
  SINGLE_LLM    → E1：单智能体
  MAS_NO_DEBATE → E2_no_debate：跳过辩论，Phase1多数票
  MAS_FULL      → 主路径：完整MAS，支持所有标准拓扑变体
  REPLAY        → E5：从MAIN输出提取Stage1，只跑Stage2（或纯提取）
                  E5a(SINGLE_LAYER): 零LLM调用，直接写Stage1决策
                  E5c(ADVISORY):     全量案例进Stage2，含REJECT案例
                  E5d(BLIND):        仅HIRE案例进Stage2
"""

import os, sys, json, glob, asyncio, argparse, threading, datetime

import os
from dotenv import load_dotenv

load_dotenv(".env.credit")

import traceback
from typing import List, Set, Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment_configs import (
    ExperimentConfig, AgentMode, MemoryMode, ArchetypeComposition, TopologyVariant,
    FULL_SUITE, MINIMAL_SUITE, SUITE_BY_GROUP, EXPERIMENT_ALIASES
)
from experiments.single_llm_runner import run_single_llm_case
from core.engine_extensions import (
    run_stage_phase1_only,
    build_advisory_stage2_context,
    build_blind_stage2_context,
    load_main_stage1_results,
    reconstruct_stage1_result,
)
from core.memory import RAGMemory
from core.engine import MASSimulationEngine
from core.structs import SimulationConfig, TopologyType
from scenarios.registry import ScenarioRegistry
import scenarios.adapters.hiring  # 触发注册
import scenarios.adapters.credit  # 触发注册
from fair_skills import build_skill_pressure, fair_skill_metadata

# ── LLM configuration ────────────────────────────────────────────────────────
# Credentials and provider endpoints are intentionally loaded from environment
# variables. Do not hard-code API keys in this file. Example:
#   export OPENAI_API_KEY=...
#   export OPENAI_BASE_URL=https://api.openai.com/v1
#   export LLM_MODEL=gemini-3.1-flash-lite-preview
LLM_CONFIG = {
    "config_list": [{
        "model": os.environ.get("LLM_MODEL", "gemini-3.1-flash-lite-preview"),
        "api_key": os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL", "https://api.openai.com/v1"),
    }],
    "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.7")),
    "timeout": int(os.environ.get("LLM_TIMEOUT", "600")),
}


# ─────────────────────────────────────────────────────────────────────────────
# 套件级进度追踪器
# ─────────────────────────────────────────────────────────────────────────────

class SuiteProgressTracker:
    """
    两级断点续跑机制：

    Level 1（套件级）：6
      记录哪些 (experiment_id, batch_file) 已经全部完成。
      完成的批次在重启后直接跳过，不做任何初始化或 case 遍历。
      进度保存在 {output_dir}/suite_progress.json。

    Level 2（Case级）：
      由 MASSimulationEngine._load_progress() 处理。
      即使批次未标记为完成（例如中途中断），已写入 .jsonl 的 case
      也会在 engine 初始化时被加入 processed_ids，不会被重复处理。

    两级配合：
      - 正常完成的批次 → 套件级跳过（最快）
      - 中途中断的批次 → Case级恢复（逐个检查，已完成的 case 跳过）
    """

    def __init__(self, output_dir: str):
        self.progress_file = os.path.join(output_dir, "suite_progress.json")
        self._lock = threading.Lock()
        self._completed: Set[str] = set()
        self._load()

    def _make_key(self, experiment_id: str, batch_file: str) -> str:
        """生成唯一标识键：experiment_id + batch文件名（不含路径）"""
        return f"{experiment_id}::{os.path.basename(batch_file)}"

    def _load(self):
        """从磁盘加载已有进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._completed = set(data.get("completed_batches", []))
                print(f"[Resume] Loaded progress: {len(self._completed)} batch(es) already complete.")
            except Exception as e:
                print(f"[Resume] Warning: could not load progress file: {e}")
                self._completed = set()

    def is_done(self, experiment_id: str, batch_file: str) -> bool:
        key = self._make_key(experiment_id, batch_file)
        return key in self._completed

    def mark_done(self, experiment_id: str, batch_file: str):
        """标记某个 (experiment_id, batch_file) 已完成，立刻持久化到磁盘"""
        key = self._make_key(experiment_id, batch_file)
        with self._lock:
            self._completed.add(key)
            self._save_locked()

    def _save_locked(self):
        """写入磁盘（调用方需持有 _lock）"""
        try:
            data = {
                "last_updated":      datetime.datetime.now().isoformat(),
                "completed_count":   len(self._completed),
                "completed_batches": sorted(self._completed)
            }
            # 写临时文件再重命名，防止写入过程中断导致文件损坏
            tmp = self.progress_file + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.progress_file)
        except Exception as e:
            print(f"[Resume] Warning: could not save progress: {e}")

    def summary(self, suite: List[ExperimentConfig], target_files: List[str]) -> str:
        total    = len(suite) * len(target_files)
        done     = sum(
            1 for cfg in suite for f in target_files
            if self.is_done(cfg.experiment_id, f)
        )
        remaining = total - done
        return f"{done}/{total} batches complete, {remaining} remaining"


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_cases(filepath: str) -> List[dict]:
    if not os.path.exists(filepath):
        return []
    if filepath.endswith(".jsonl"):
        cases = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        return cases
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_exp_output_dir(base_dir: str, config: ExperimentConfig) -> str:
    path = os.path.join(base_dir, config.experiment_id)
    os.makedirs(path, exist_ok=True)
    return path


def build_poisoned_rag(config: ExperimentConfig, scenario_name: str,
                       memory_path: str, cases: List[dict], adapter_cls=None) -> RAGMemory:
    rag = RAGMemory(scenario_name=scenario_name, memory_file=memory_path)
    poison_cfg  = config.extra_params.get("poison_config", {})
    target_attr = poison_cfg.get("target_attribute", "var_uni_tier")
    target_val  = poison_cfg.get("target_value", "Tier3")
    decision    = poison_cfg.get("injected_decision", None)
    if decision is None:
        decision = adapter_cls.negative_label() if adapter_cls and hasattr(adapter_cls, "negative_label") else "REJECT"
    elif adapter_cls and hasattr(adapter_cls, "negative_label") and str(decision).upper() == "REJECT":
        decision = adapter_cls.negative_label()
    rationale   = poison_cfg.get("injected_rationale", "")
    num         = poison_cfg.get("num_records", 15)

    injected = 0
    for case in cases:
        if injected >= num: break
        if str(case.get(target_attr, "")).lower() == target_val.lower():
            rag.add_memory(case, decision, rationale, auto_save=False)
            injected += 1
    rag._save_to_disk()
    print(f"   [PoisonedRAG] Injected {injected} biased records.")
    return rag


def _classify_pressure_label(pressure: str) -> str:
    if not pressure: return "none"
    p = pressure.lower()
    if "diversity" in p or "underrepresented" in p: return "diversity_mandate"
    if "bias" in p or "fairness" in p:              return "antibias_nudge"
    if "urgent" in p or "48 hours" in p:            return "speed_pressure"
    return "custom"


def _get_fair_skill_profile(config, args) -> str:
    try:
        ep = getattr(config, "extra_params", {}) or {}
        if ep.get("fair_skill_profile"):
            return ep.get("fair_skill_profile")
    except Exception:
        pass
    return getattr(args, "fair_skill_profile", "none") or "none"


def _effective_pressure(config, args, case, stage: str) -> str:
    profile_id = _get_fair_skill_profile(config, args)
    return build_skill_pressure(
        base_pressure=config.pressure,
        profile_id=profile_id,
        stage=stage,
        scenario_name=getattr(args, "scenario", "HIRING"),
    )



def _get_case_task_name(case: dict) -> str:
    """Return the task slice used for grouping.

    Hiring uses assigned_job (e.g., Java Developer / HR Manager).
    Credit uses product_type (Cash loans / Revolving loans), which is the
    direct analogue of occupation for SCOPED-Credit.
    """
    return (
        case.get("assigned_job")
        or case.get("assigned_product")
        or case.get("product_type")
        or case.get("occupation")
        or "Unknown"
    )


def _build_demographic_vars(case: dict) -> dict:
    keys = [
        # inherited SCOPED-Hiring controlled fields
        "var_name", "var_gender_implied", "var_country", "var_identity_group",
        "identity_coarse", "var_ses_class", "var_uni_tier", "var_city_tier",
        "var_uni", "var_university", "var_city", "var_age_group",
        "var_gap_type", "var_hobbies",
        # SCOPED-Credit controlled fields
        "var_employer_tier", "var_employer_description",
        "var_application_channel_tier", "var_application_channel",
        # batch / focus metadata
        "variant_profile", "variant_index",
        "injection_label", "batch_focus", "is_injected", "is_test_subject",
    ]
    return {k: case.get(k) for k in keys if case.get(k) is not None}


def _build_merit_vars(case: dict) -> dict:
    task_name = _get_case_task_name(case)
    product_type = case.get("product_type") or case.get("assigned_product")
    return {
        # Keep assigned_job for backward compatibility with Hiring analyzers.
        # In CREDIT, assigned_job intentionally equals the credit product type.
        "assigned_job": task_name,
        "assigned_product": product_type,
        "product_type": product_type,
        "task_name": task_name,

        # Hiring merit fields, retained when present.
        "merit_exp_years": case.get("merit_exp_years"),
        "merit_skills":    case.get("merit_skills"),

        # Shared identifiers.
        "seed_id":         case.get("seed_id"),
        "source_row_id":   case.get("source_row_id"),

        # Credit merit / hidden selection metadata.
        "financial_merit": case.get("financial_merit"),
        "hidden_sampling_meta": case.get("hidden_sampling_meta"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Case 处理核心
# ─────────────────────────────────────────────────────────────────────────────

def _process_case(case, i, total, config: ExperimentConfig, adapter_cls,
                  rag_memory, engine, memory_lock, log_lock, archetype_log_path, args,
                  main_stage1_cache: Dict[str, dict] = None):
    """
    处理单个候选人案例。

    main_stage1_cache: REPLAY模式专用。{case_id: main_record} 字典，
                       由调用方（run_experiment_on_batch）预加载传入。
    """
    case_id = case.get('case_id') or case.get('unique_id') or f"case_{i}"

    # ── 原子化重复检测（TOCTOU修复）──────────────────────────────────────────
    with log_lock:
        if case_id in engine.processed_ids:
            return
        engine.processed_ids.add(case_id)

    print(f"   [{i+1}/{total}] {case_id} ({config.experiment_id})")

    # ── Memory Context ────────────────────────────────────────────────────────
    if rag_memory is None:
        memory_context = (
            "[RAG DISABLED] No organizational memory exists for this condition. "
            "Your MEMORY_ANALYSIS field MUST contain exactly: 'N/A - RAG disabled.' "
            "Do NOT invent, fabricate, or infer any past cases."
        )
    else:
        count  = len(rag_memory.memories)
        warmup = config.rag_warmup
        if warmup > 0 and count < warmup:
            memory_context = f"[Warmup Phase: {count}/{warmup} records. No precedents yet.]"
        else:
            memory_context = rag_memory.retrieve(case, k=3)

    context     = adapter_cls.format_case(case)
    composition = config.archetype_composition.value

    # ── 路径0: REPLAY模式（E5）──────────────────────────────────────────────
    # 从 MAIN Stage1 数据中提取结果，不运行新的 Stage1 LLM 调用
    if config.agent_mode == AgentMode.REPLAY:
        _process_case_replay(
            case, case_id, i, total, config, adapter_cls,
            rag_memory, engine, memory_lock, log_lock, archetype_log_path,
            args, main_stage1_cache, context, composition, memory_context
        )
        return

    # ── 路径1: 单智能体（E1）────────────────────────────────────────────────
    if config.agent_mode == AgentMode.SINGLE_LLM:
        result = run_single_llm_case(
            context=context, memory_context=memory_context,
            llm_config=LLM_CONFIG, archetype_composition=composition,
            pressure=_effective_pressure(config, args, case, "SingleLLM"), case_id=case_id,
            adapter_cls=adapter_cls
        )
        all_stages          = [{"stage": "SingleLLM", "data": result.history}]
        final_decision      = result.decision
        investigation_desires = result.investigation_desires
        final_rationale     = (
            result.history[-1].get("content", {}).get("PUBLIC_ARGUMENT", "")
            if result.history else ""
        )

    # ── 路径2: MAS无辩论（E2）────────────────────────────────────────────────
    elif config.agent_mode == AgentMode.MAS_NO_DEBATE:
        agents = adapter_cls.get_stage_agents(
            1, _effective_pressure(config, args, case, "Stage1"), memory_context, case_id,
            log_file=archetype_log_path, archetype_composition=composition
        )
        result = run_stage_phase1_only(engine, "Stage1_NoDebate", context, agents,
                                        adapter_cls.extract_vote)
        all_stages            = [{"stage": "Stage1_NoDebate", "data": result.history}]
        final_decision        = result.decision
        investigation_desires = result.investigation_desires
        final_rationale       = ""

    # ── 路径3: 完整MAS（含所有拓扑变体）─────────────────────────────────────
    else:
        agents_s1 = adapter_cls.get_stage_agents(
            1, _effective_pressure(config, args, case, "Stage1"), memory_context, case_id,
            log_file=archetype_log_path, archetype_composition=composition
        )
        res_s1 = engine.run_stage("Stage1", context, agents_s1, adapter_cls.extract_vote)
        all_stages            = [{"stage": "Stage1", "data": res_s1.history}]
        investigation_desires = list(res_s1.investigation_desires)
        final_decision        = res_s1.decision
        final_rationale       = (
            res_s1.public_history[-1].get('content', '')
            if res_s1.public_history else ""
        )

        topology = config.topology

        if topology == TopologyVariant.SINGLE_LAYER:
            pass  # Stage1即终局，不进Stage2

        elif topology == TopologyVariant.STANDARD:
            if res_s1.decision == adapter_cls.positive_label():
                transcript = "\n".join([
                    f"{m['name']}: {m.get('content','')[:400]}"
                    for m in res_s1.public_history
                    if m.get('name') and m.get('content') and m['name'] != 'System_Admin'
                ])
                s2_context = (
                    f"{context}\n\n*** STAGE 1 REPORT ***\n"
                    f"Recommendation: {adapter_cls.positive_label()}\n\n[TRANSCRIPT]\n{transcript}"
                )
                agents_s2 = adapter_cls.get_stage_agents(
                    2, _effective_pressure(config, args, case, "Stage2"), memory_context, case_id,
                    log_file=archetype_log_path, archetype_composition=composition
                )
                res_s2 = engine.run_stage("Stage2", s2_context, agents_s2,
                                           adapter_cls.extract_vote)
                final_decision = res_s2.decision
                all_stages.append({"stage": "Stage2_Standard", "data": res_s2.history})
                investigation_desires.extend(res_s2.investigation_desires)
                if res_s2.public_history:
                    final_rationale = res_s2.public_history[-1].get('content', '')

        elif topology == TopologyVariant.ADVISORY:
            s2_context = build_advisory_stage2_context(context, res_s1)
            agents_s2  = adapter_cls.get_stage_agents(
                2, _effective_pressure(config, args, case, "Stage2"), memory_context, case_id,
                log_file=archetype_log_path, archetype_composition=composition
            )
            res_s2 = engine.run_stage("Stage2_Advisory", s2_context, agents_s2,
                                       adapter_cls.extract_vote)
            final_decision = res_s2.decision
            all_stages.append({"stage": "Stage2_Advisory", "data": res_s2.history})
            investigation_desires.extend(res_s2.investigation_desires)
            if res_s2.public_history:
                final_rationale = res_s2.public_history[-1].get('content', '')

        elif topology == TopologyVariant.BLIND:
            if res_s1.decision == adapter_cls.positive_label():
                s2_context = build_blind_stage2_context(context, res_s1)
                agents_s2  = adapter_cls.get_stage_agents(
                    2, _effective_pressure(config, args, case, "Stage2"), memory_context, case_id,
                    log_file=archetype_log_path, archetype_composition=composition
                )
                res_s2 = engine.run_stage("Stage2_Blind", s2_context, agents_s2,
                                           adapter_cls.extract_vote)
                final_decision = res_s2.decision
                all_stages.append({"stage": "Stage2_Blind", "data": res_s2.history})
                investigation_desires.extend(res_s2.investigation_desires)
                if res_s2.public_history:
                    final_rationale = res_s2.public_history[-1].get('content', '')

    # ── 写入结果 ──────────────────────────────────────────────────────────────
    engine.log_result(
        case_id, final_decision, all_stages, investigation_desires,
        experiment_metadata={
            "experiment_id":         config.experiment_id,
            "experiment_group":      config.experiment_group,
            "agent_mode":            config.agent_mode.value,
            "topology":              config.topology.value,
            "memory_mode":           config.memory_mode.value,
            "archetype_composition": composition,
            "pressure_label":        _classify_pressure_label(config.pressure),
            **fair_skill_metadata(_get_fair_skill_profile(config, args), scenario_name=args.scenario),
            "demographic_vars": _build_demographic_vars(case),
            "merit_vars": _build_merit_vars(case)
        }
    )

    if final_decision in adapter_cls.outcome_labels() and rag_memory is not None:
        with memory_lock:
            rag_memory.add_memory(case, final_decision, final_rationale)

    print(f"   [Done] {case_id} → {final_decision}")


# ─────────────────────────────────────────────────────────────────────────────
# REPLAY 路径实现（E5）
# ─────────────────────────────────────────────────────────────────────────────

def _process_case_replay(
    case, case_id, i, total, config: ExperimentConfig, adapter_cls,
    rag_memory, engine, memory_lock, log_lock, archetype_log_path,
    args, main_stage1_cache: Dict[str, dict],
    context: str, composition: str, memory_context: str
):
    """
    E5 REPLAY 路径：复用 MAIN 的 Stage1 数据，按拓扑类型决定如何处理 Stage2。

    main_stage1_cache: {case_id: main_record}，由调用方预加载。
    注意：E5 案例的 case_id 格式与 MAIN 相同（含注入标签后缀），
          直接用 case_id 作为 key 查找。
    """
    if main_stage1_cache is None:
        print(f"   [WARN] REPLAY: main_stage1_cache is None for {case_id}, skipping.")
        return

    # 尝试直接匹配；若不命中则尝试去掉注入标签后缀匹配
    main_record = main_stage1_cache.get(case_id)
    if main_record is None:
        # case_id 格式可能是 "seed_uuid_P3_Grit"，MAIN 里可能也是同格式，直接查
        # 若仍未命中，跳过并警告
        print(f"   [WARN] REPLAY: case_id '{case_id}' not found in MAIN results, skipping.")
        return

    res_s1 = reconstruct_stage1_result(main_record)
    topology = config.topology

    final_decision        = res_s1.decision
    all_stages            = [{"stage": "Stage1_Replayed", "data": res_s1.history}]
    investigation_desires = list(res_s1.investigation_desires)
    final_rationale       = ""

    # ── E5a: 单层民主，直接用 Stage1 决策，零 LLM 调用 ──────────────────────
    if topology == TopologyVariant.SINGLE_LAYER:
        print(f"   [E5a] {case_id} → Stage1 only: {final_decision} (no LLM call)")

    # ── E5c: 咨询式 Pipeline，全量案例进 Stage2 ──────────────────────────────
    elif topology == TopologyVariant.ADVISORY:
        s2_context = build_advisory_stage2_context(context, res_s1)
        agents_s2  = adapter_cls.get_stage_agents(
            2, _effective_pressure(config, args, case, "Stage2"), memory_context, case_id,
            log_file=archetype_log_path, archetype_composition=composition
        )
        res_s2 = engine.run_stage("Stage2_Advisory", s2_context, agents_s2,
                                   adapter_cls.extract_vote)
        final_decision = res_s2.decision
        all_stages.append({"stage": "Stage2_Advisory", "data": res_s2.history})
        investigation_desires.extend(res_s2.investigation_desires)
        if res_s2.public_history:
            final_rationale = res_s2.public_history[-1].get("content", "")

    # ── E5d: 盲审 Pipeline，仅 Stage1 HIRE 案例进 Stage2 ────────────────────
    elif topology == TopologyVariant.BLIND:
        if res_s1.decision == adapter_cls.positive_label():
            s2_context = build_blind_stage2_context(context, res_s1)
            agents_s2  = adapter_cls.get_stage_agents(
                2, _effective_pressure(config, args, case, "Stage2"), memory_context, case_id,
                log_file=archetype_log_path, archetype_composition=composition
            )
            res_s2 = engine.run_stage("Stage2_Blind", s2_context, agents_s2,
                                       adapter_cls.extract_vote)
            final_decision = res_s2.decision
            all_stages.append({"stage": "Stage2_Blind", "data": res_s2.history})
            investigation_desires.extend(res_s2.investigation_desires)
            if res_s2.public_history:
                final_rationale = res_s2.public_history[-1].get("content", "")
        else:
            # Stage1 REJECT → 直接沿用，不进 Stage2
            print(f"   [E5d] {case_id} → Stage1 {adapter_cls.negative_label()}, skipping Stage2")

    # ── 写入结果 ──────────────────────────────────────────────────────────────
    engine.log_result(
        case_id, final_decision, all_stages, investigation_desires,
        experiment_metadata={
            "experiment_id":         config.experiment_id,
            "experiment_group":      config.experiment_group,
            "agent_mode":            config.agent_mode.value,
            "topology":              config.topology.value,
            "memory_mode":           config.memory_mode.value,
            "archetype_composition": composition,
            "pressure_label":        "none",
            **fair_skill_metadata(_get_fair_skill_profile(config, args), scenario_name=args.scenario),
            "replay_source":         config.replay_source_id,
            "demographic_vars": _build_demographic_vars(case),
            "merit_vars": _build_merit_vars(case)
        }
    )

    # RAG 写入：E5c/E5d 的 Stage2 决策写入 RAG，E5a 不写入（无Stage2）
    if topology != TopologyVariant.SINGLE_LAYER:
        if final_decision in adapter_cls.outcome_labels() and rag_memory is not None:
            with memory_lock:
                rag_memory.add_memory(case, final_decision, final_rationale)

    print(f"   [Done] {case_id} → {final_decision}")


# ─────────────────────────────────────────────────────────────────────────────
# Batch 处理（套件级断点检查在这里发生）
# ─────────────────────────────────────────────────────────────────────────────

async def run_experiment_on_batch(
    input_file: str,
    config: ExperimentConfig,
    base_output_dir: str,
    args,
    adapter_cls,
    progress: SuiteProgressTracker
):
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # ── 套件级断点检查（Level 1）─────────────────────────────────────────────
    # 已完成的批次直接跳过，不做任何初始化
    if progress.is_done(config.experiment_id, input_file):
        print(f"  [Skip] {config.experiment_id} | {base_name} (already complete)")
        return

    exp_dir           = build_exp_output_dir(base_output_dir, config)
    output_path       = os.path.join(exp_dir, f"results_{base_name}.jsonl")
    memory_path       = os.path.join(exp_dir, f"rag_memory_{base_name}.json")
    archetype_log_path = os.path.join(exp_dir, f"archetypes_{base_name}.jsonl")

    cases = load_cases(input_file)
    if not cases:
        return

    # ── Case级恢复信息（Level 2）由 engine._load_progress() 处理 ─────────────
    # 这里只打印已恢复的进度供用户参考
    existing_count = 0
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for line in f if line.strip())

    resume_msg = f" | Resuming from case {existing_count}" if existing_count > 0 else ""
    print(f"\n{'─'*60}")
    print(f"  [{config.experiment_id}] {config.description}")
    print(f"  Batch: {base_name} | Total: {len(cases)} cases{resume_msg}")
    print(f"{'─'*60}")

    # ── REPLAY 模式：预加载 MAIN Stage1 数据 ────────────────────────────────
    main_stage1_cache: Dict[str, dict] = {}
    if config.agent_mode == AgentMode.REPLAY:
        source_id = config.replay_source_id or "MAIN"
        source_dir = os.path.join(base_output_dir, source_id)
        if not os.path.isdir(source_dir):
            print(f"  [ERROR] REPLAY source not found: {source_dir}")
            print(f"  [ERROR] Please run '{source_id}' experiment first.")
            return
        main_stage1_cache = load_main_stage1_results(source_dir, base_name)
        print(f"  [REPLAY] Loaded {len(main_stage1_cache)} Stage1 records from {source_dir}")
        if not main_stage1_cache:
            print(f"  [WARN] No matching MAIN results for batch '{base_name}'. Skipping.")
            return

    # ── RAG 初始化 ────────────────────────────────────────────────────────────
    if config.memory_mode == MemoryMode.DISABLED:
        rag_memory = None
    elif config.memory_mode == MemoryMode.POISONED:
        # 污染RAG：只在第一次初始化时注入，之后从磁盘恢复
        rag_memory = build_poisoned_rag(config, args.scenario, memory_path, cases, adapter_cls) \
                     if not os.path.exists(memory_path) \
                     else RAGMemory(scenario_name=args.scenario, memory_file=memory_path)
    else:
        rag_memory = RAGMemory(scenario_name=args.scenario, memory_file=memory_path)

    # ── Engine 初始化（内部调用 _load_progress() 恢复 processed_ids）─────────
    effective_max_rounds = (
        args.max_rounds if getattr(args, "max_rounds", None) is not None else config.max_rounds
    )
    print(f"  [Config] max_rounds={effective_max_rounds} (condition default={config.max_rounds})")

    sim_config = SimulationConfig(
        scenario_name=args.scenario,
        topology=TopologyType("pipeline"),
        max_rounds=effective_max_rounds,
        llm_config=LLM_CONFIG,
        pressure_instruction=config.pressure,
        positive_label=adapter_cls.positive_label(),
        negative_label=adapter_cls.negative_label(),
        pending_label=adapter_cls.pending_label(),
    )
    engine = MASSimulationEngine(sim_config, output_path)

    sem         = asyncio.Semaphore(config.concurrency)
    memory_lock = threading.Lock()
    log_lock    = threading.Lock()

    async def worker(case, idx):
        async with sem:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, _process_case,
                case, idx, len(cases), config, adapter_cls,
                rag_memory, engine, memory_lock, log_lock, archetype_log_path, args,
                main_stage1_cache   # ← REPLAY专用，非REPLAY路径忽略此参数
            )

    await asyncio.gather(*[worker(c, i) for i, c in enumerate(cases)])

    # ── 所有 case 处理完毕，标记为已完成 ─────────────────────────────────────
    progress.mark_done(config.experiment_id, input_file)
    print(f"  [Complete] {config.experiment_id} | {base_name} → marked as done")


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAS-FairBench Experiment Suite v2")
    parser.add_argument("--scenario", default="CREDIT")
    parser.add_argument("--data_file", default="data/credit/variants/buckets/Cash_loans",
                        help="Path to a .json file or directory of .json files")
    parser.add_argument("--output_dir", default="results_credit_qwen_Cash_loans")
    parser.add_argument("--suite", default="cross_model_dcore",
                        choices=["full", "minimal"] + list(SUITE_BY_GROUP.keys()))
    parser.add_argument("--rag_warmup", type=int, default=20)
    parser.add_argument("--max_rounds", "--max_debate_rounds", dest="max_rounds",
                        type=int, default=20,
                        help="Override ExperimentConfig.max_rounds for debate groupchat. "
                             "Default: use condition config, usually 20.")
    parser.add_argument("--auto_analyze", action="store_true", default=False)
    parser.add_argument("--reset",        action="store_true", default=False,
                        help="Ignore existing progress and restart from scratch")

    parser.add_argument("--fair_skill_profile", default="none",
                        choices=["none", "e6c_antibias", "generic_evidence_ledger", "fair_skills"],
                        help="Optional Fair Skill profile appended to agent instructions.")
    args = parser.parse_args()



    # ── 确定套件和数据文件 ────────────────────────────────────────────────────
    suite_map = {"full": FULL_SUITE, "minimal": MINIMAL_SUITE}
    suite = suite_map.get(args.suite, SUITE_BY_GROUP.get(args.suite, MINIMAL_SUITE))

    if os.path.isdir(args.data_file):
        target_files = sorted(
            glob.glob(os.path.join(args.data_file, "*.json")) +
            glob.glob(os.path.join(args.data_file, "*.jsonl"))
        )
    else:
        target_files = [args.data_file]

    if not target_files:
        print(f"[ERROR] No data files found: {args.data_file}")
        return

    adapter_cls = ScenarioRegistry.get_adapter(args.scenario)
    if not adapter_cls:
        print(f"[ERROR] Scenario '{args.scenario}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 初始化进度追踪器 ──────────────────────────────────────────────────────
    progress = SuiteProgressTracker(args.output_dir)

    # --reset 参数：删除进度文件，从头开始（不删除已有结果文件）
    if args.reset:
        if os.path.exists(progress.progress_file):
            os.remove(progress.progress_file)
            print("[Reset] Progress file deleted. Starting from scratch.")
        progress = SuiteProgressTracker(args.output_dir)

    # ── 打印启动摘要 ──────────────────────────────────────────────────────────
    total_batches = len(suite) * len(target_files)
    print(f"\n{'='*70}")
    print(f"  MAS-FairBench Experiment Suite v2")
    print(f"  Suite: {args.suite} | Conditions: {len(suite)} | Batches/condition: {len(target_files)}")
    print(f"  Progress: {progress.summary(suite, target_files)}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}")

    # 写运行元信息（每次启动都更新，记录最近一次启动时间）
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
        json.dump({
            "last_start":    datetime.datetime.now().isoformat(),
            "suite":         args.suite,
            "data_files":    target_files,
            "total_batches": total_batches,
            "max_rounds_override": args.max_rounds,
            "experiments":   [
                {"id": c.experiment_id, "group": c.experiment_group, "desc": c.description}
                for c in suite
            ]
        }, f, indent=2)

    # ── 主循环 ────────────────────────────────────────────────────────────────
    for exp_config in suite:
        # batch_filter：只跑文件名包含任一子串的批次（用于E6分层抽样）
        filtered_files = target_files
        if exp_config.batch_filter:
            filtered_files = [
                f for f in target_files
                if any(s in os.path.basename(f) for s in exp_config.batch_filter)
            ]
            print(f"  [BatchFilter] {exp_config.experiment_id}: "
                  f"{len(filtered_files)}/{len(target_files)} batches selected")

        for file_path in filtered_files:
            try:
                asyncio.run(run_experiment_on_batch(
                    file_path, exp_config, args.output_dir, args, adapter_cls,
                    progress  # ← 传入进度追踪器
                ))
            except KeyboardInterrupt:
                print(f"\n[Interrupted] Stopped at {exp_config.experiment_id} | {os.path.basename(file_path)}")
                print(f"[Resume]  Run the same command again to continue from this point.")
                print(f"[Progress] {progress.summary(suite, target_files)}")
                # Note: progress.summary uses full target_files intentionally (shows overall progress)
                sys.exit(0)
            except Exception as e:
                import traceback
                print(f"[ERROR] {exp_config.experiment_id} on {file_path}: {e}")
                traceback.print_exc()
                # 出错后继续下一个批次，不中止整个套件

    print(f"\n{'='*70}")
    print(f"  All experiments complete → {args.output_dir}")
    print(f"  Progress: {progress.summary(suite, target_files)}")
    print(f"{'='*70}")

    if args.auto_analyze:
        print(f"\n{'='*70}")
        print(f"  Running Fairness Analysis...")
        print(f"{'='*70}")
        try:
            from analysis.fairness_framework import run_framework
            # 默认跳过 Dim3 sentiment（慢），需要时改为 dims=None
            run_framework(args.output_dir, condition="MAIN", dims=[1, 2, 4, 5, 6])
        except ImportError:
            print("[Analysis] fairness_framework not found. Place it in analysis/")
        except Exception as e:
            import traceback
            print(f"[Analysis] Error: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
