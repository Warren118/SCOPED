# fair_skills/bindings/hiring.py
# -*- coding: utf-8 -*-
"""
Hiring scenario binding for domain-general Fair Skills.

This is the adapter layer:
generic mechanism skill -> hiring-specific examples/fields -> rendered MAS prompt.
"""

from __future__ import annotations

from ..core import ScenarioBinding


HIRING_BINDING = ScenarioBinding(
    scenario_id="hiring",
    scenario_label="LLM-based multi-agent hiring decision",
    decision_unit="candidate",
    outcome_labels=["HIRE", "REJECT"],
    score_scale="1-10 hiring rubric",
    score_names=["Competency", "Potential", "Fit", "Value", "Risk"],
    direct_evidence_examples=[
        "demonstrated skills",
        "projects",
        "work outcomes",
        "role-relevant achievements",
        "responsibilities",
        "internships",
        "research or technical artifacts",
        "experience years",
        "concrete gaps in required capabilities",
    ],
    uncertainty_examples=[
        "career gap",
        "nonlinear career path",
        "missing employment dates",
        "ambiguous timeline",
        "unfamiliar institution",
        "incomplete evidence",
    ],
    proxy_examples=[
        "university prestige",
        "university tier",
        "unknown or lower-tier institution",
        "employer prestige",
        "city tier",
        "country familiarity",
        "name familiarity",
        "hobbies as socioeconomic cues",
        "background familiarity",
    ],
    investigation_examples=[
        "degree verification",
        "employment-history verification",
        "GitHub or portfolio verification",
        "communication/culture concern",
        "credential credibility concern",
        "background check request",
    ],
    identity_background_examples=[
        "name",
        "country",
        "language",
        "identity group",
        "culture fit",
        "communication style",
        "unfamiliar background",
    ],
    extra_notes=(
        "In hiring, a school or employer name alone is a credential/proxy label, not demonstrated ability. "
        "Education or work history may be used only when the résumé describes concrete task-relevant evidence."
    ),
)
