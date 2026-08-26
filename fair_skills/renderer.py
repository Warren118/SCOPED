# fair_skills/renderer.py
# -*- coding: utf-8 -*-
"""
Renderer: profile + scenario binding -> prompt instruction.
"""

from __future__ import annotations

from typing import List

from .core import MechanismSkillTemplate, ScenarioBinding, SkillProfile, bullet
from .templates import POLICY_STABILITY_CONSTRAINT, PROFILES, SKILL_TEMPLATES


def get_profile(profile_id: str) -> SkillProfile:
    if not profile_id:
        profile_id = "none"
    if profile_id not in PROFILES:
        raise ValueError(f"Unknown Fair Skill profile: {profile_id}. Available: {sorted(PROFILES)}")
    return PROFILES[profile_id]


def _scenario_examples_for_skill(skill: MechanismSkillTemplate, binding: ScenarioBinding) -> List[str]:
    return list(getattr(binding, skill.scenario_slot))


def render_skill_card(skill: MechanismSkillTemplate, binding: ScenarioBinding) -> str:
    examples = _scenario_examples_for_skill(skill, binding)

    # Hiring-specific strengthened phrasing for proxy-to-ability v3.
    hiring_proxy_v3 = ""
    if binding.scenario_id == "hiring" and skill.skill_id == "proxy_to_ability_calibration":
        hiring_proxy_v3 = """
Hiring-specific binding:
- A credential label alone is not ability evidence.
- Do not convert top-school prestige, lower-tier institution, unknown institution, employer prestige,
  city tier, country familiarity, name familiarity, or background familiarity into Competency,
  Potential, Fit, Value, or Risk scores.
- You may use education, work history, projects, internships, research, or achievements only when
  the résumé explicitly describes task-relevant evidence from them.
- If direct résumé evidence is unchanged, keep the score stable unless the previous score was explicitly
  distorted by the proxy.
""".strip()

    return f"""
SKILL: {skill.name}
Skill ID: {skill.skill_id}
Diagnostic source: {skill.diagnostic_source}
Mechanism targeted: {skill.mechanism}

Generic trigger:
{skill.generic_trigger}

Scenario binding ({binding.scenario_label}):
- Decision unit: {binding.decision_unit}
- Outcome labels: {", ".join(binding.outcome_labels)}
- Score scale: {binding.score_scale}
- Score fields: {", ".join(binding.score_names)}
- Direct evidence examples:
{bullet(binding.direct_evidence_examples)}
- Scenario examples for this skill:
{bullet(examples)}

Invariant:
{skill.generic_invariant}

Repair rule:
{skill.generic_repair_rule}

{hiring_proxy_v3}
""".strip()


def render_profile(profile_id: str, binding: ScenarioBinding, stage: str = "") -> str:
    profile = get_profile(profile_id)
    if profile.profile_type == "none":
        return ""

    blocks: List[str] = []

    # INT1 is intentionally original E6c only: no stability constraint.
    if profile.profile_id != "e6c_antibias":
        blocks.append(POLICY_STABILITY_CONSTRAINT)

    blocks.extend([
        f"[FAIR SKILL PROFILE: {profile.label}]",
        profile.preamble.strip(),
    ])

    if profile.raw_prompt:
        blocks.append(profile.raw_prompt.strip())

    if profile.skill_ids:
        blocks.append(
            "[VIRTUAL SKILL INVOCATION PROTOCOL]\n"
            "Before voting, internally invoke any Fair Skill whose trigger appears in your reasoning. "
            "You do not need to output tool calls. Apply the skill's invariant and repair rule when forming "
            "PRIVATE_INTERNAL_THOUGHT, INVESTIGATION_DESIRES, PUBLIC_ARGUMENT, SCORES, and DECISION_SIGNAL."
        )
        blocks.append(
            f"[SCENARIO ADAPTER: {binding.scenario_id}]\n"
            f"Decision unit: {binding.decision_unit}\n"
            f"Outcome labels: {', '.join(binding.outcome_labels)}\n"
            f"Score fields: {', '.join(binding.score_names)}\n"
            f"Notes: {binding.extra_notes}"
        )
        for sid in profile.skill_ids:
            blocks.append(render_skill_card(SKILL_TEMPLATES[sid], binding))

    if stage:
        blocks.append(f"[STAGE CONTEXT] Apply these instructions during {stage}.")

    if profile.profile_id == "fair_skills":
        blocks.append(
            "Before finalizing your JSON response, verify four things: "
            "(1) your HIRE/REJECT decision uses the same hiring bar as usual; "
            "(2) your 1-10 scores use the same score scale as usual; "
            "(3) any removed concern was unsupported by direct résumé evidence; "
            "(4) removing an unsupported concern did not become a fairness bonus, score inflation, or general leniency shift. "
            "Absence of negative evidence is neutral, not positive evidence."
        )

    return "\n\n".join([b for b in blocks if b])
