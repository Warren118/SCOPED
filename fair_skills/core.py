# fair_skills/core.py
# -*- coding: utf-8 -*-
"""
Core data structures for virtual Fair Skills.

Design:
- MechanismSkillTemplate: domain-general mechanism skill.
- ScenarioBinding: scenario-specific grounding of generic skill triggers/examples.
- SkillProfile: runtime profile used by intervention conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MechanismSkillTemplate:
    skill_id: str
    name: str
    diagnostic_source: str
    mechanism: str
    generic_trigger: str
    generic_invariant: str
    generic_repair_rule: str
    scenario_slot: str
    proximal_metrics: List[str]
    spillover_metrics: List[str]


@dataclass(frozen=True)
class ScenarioBinding:
    scenario_id: str
    scenario_label: str
    decision_unit: str
    outcome_labels: List[str]
    score_scale: str
    score_names: List[str]
    direct_evidence_examples: List[str]
    uncertainty_examples: List[str]
    proxy_examples: List[str]
    investigation_examples: List[str]
    identity_background_examples: List[str]
    extra_notes: str = ""


@dataclass(frozen=True)
class SkillProfile:
    profile_id: str
    label: str
    description: str
    profile_type: str  # none | generic_baseline | fair_skills
    skill_ids: List[str]
    preamble: str = ""
    raw_prompt: str = ""


def bullet(items: List[str]) -> str:
    return "\n".join([f"- {x}" for x in items])
