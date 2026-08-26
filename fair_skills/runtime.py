# fair_skills/runtime.py
# -*- coding: utf-8 -*-
"""Runtime compatibility API expected by run_suite.py."""

from __future__ import annotations

from .bindings.hiring import HIRING_BINDING
from .bindings.credit import CREDIT_BINDING
from .renderer import get_profile, render_profile
from .templates import SKILL_TEMPLATES

BINDINGS = {
    "HIRING": HIRING_BINDING,
    "hiring": HIRING_BINDING,
    "CREDIT": CREDIT_BINDING,
    "credit": CREDIT_BINDING,
}


def get_scenario_binding(scenario_name: str = "HIRING"):
    return BINDINGS.get(str(scenario_name), BINDINGS.get(str(scenario_name).upper(), HIRING_BINDING))


def build_skill_pressure(
    base_pressure: str = "",
    profile_id: str = "none",
    stage: str = "",
    scenario_name: str = "HIRING",
    scenario_binding=None,
) -> str:
    binding = scenario_binding or get_scenario_binding(scenario_name)
    parts = []
    if base_pressure and str(base_pressure).strip():
        parts.append(str(base_pressure).strip())

    rendered = render_profile(profile_id, binding, stage=stage)
    if rendered.strip():
        parts.append(rendered.strip())

    return "\n\n".join(parts).strip()


def fair_skill_metadata(profile_id: str, scenario_name: str = "HIRING") -> dict:
    binding = get_scenario_binding(scenario_name)
    profile = get_profile(profile_id)
    skill_templates = [SKILL_TEMPLATES[sid] for sid in profile.skill_ids]

    return {
        "fair_skill_profile": profile.profile_id,
        "fair_skill_label": profile.label,
        "fair_skill_description": profile.description,
        "fair_skill_profile_type": profile.profile_type,
        "fair_skill_ids": [s.skill_id for s in skill_templates],
        "fair_skill_names": [s.name for s in skill_templates],
        "fair_skill_diagnostic_sources": [s.diagnostic_source for s in skill_templates],
        "fair_skill_mechanisms": [s.mechanism for s in skill_templates],
        "fair_skill_triggers": [s.generic_trigger for s in skill_templates],
        "fair_skill_invariants": [s.generic_invariant for s in skill_templates],
        "scenario_binding": binding.scenario_id,
        "virtual_skill_invocation": bool(profile.skill_ids),
        "policy_stability_constraint": bool(profile.profile_id not in ["none", "e6c_antibias"]),
    }


def build_fair_skill_instruction(profile_id: str, stage: str = "", scenario_name: str = "HIRING") -> str:
    return render_profile(profile_id, get_scenario_binding(scenario_name), stage=stage)
