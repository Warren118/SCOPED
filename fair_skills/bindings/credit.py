# fair_skills/bindings/credit.py
# -*- coding: utf-8 -*-
"""Credit scenario binding for domain-general Fair Skills."""

from __future__ import annotations

from ..core import ScenarioBinding


CREDIT_BINDING = ScenarioBinding(
    scenario_id="credit",
    scenario_label="LLM-based multi-agent credit approval decision",
    decision_unit="loan application",
    outcome_labels=["APPROVE", "DENY"],
    score_scale="1-10 credit underwriting rubric",
    score_names=[
        "Repayment_Capacity",
        "Income_Stability",
        "Documentation_Strength",
        "Portfolio_Value",
        "Default_Risk",
        "Compliance_Risk",
    ],
    direct_evidence_examples=[
        "declared verified income",
        "requested credit amount",
        "annuity burden",
        "credit-to-income ratio",
        "annuity-to-income ratio",
        "employment duration in current income source",
        "income type",
        "household size and dependents",
        "supporting document count",
        "recent credit inquiry count",
        "housing and asset context",
    ],
    uncertainty_examples=[
        "prior income interruption",
        "contract income transition",
        "missing narrative details",
        "manual application intake",
        "unfamiliar institution or employer label",
        "ambiguous documentation concern",
        "non-standard application channel",
    ],
    proxy_examples=[
        "university prestige",
        "university tier",
        "employer prestige",
        "city tier",
        "country familiarity",
        "name familiarity",
        "hobbies as socioeconomic cues",
        "application channel",
        "community partner channel",
        "background familiarity",
    ],
    investigation_examples=[
        "income verification",
        "employment verification",
        "bank-statement request",
        "KYC review",
        "manual document review",
        "residence check",
        "credibility concern",
    ],
    identity_background_examples=[
        "name",
        "country",
        "language",
        "identity group",
        "city familiarity",
        "school familiarity",
        "application channel",
        "unfamiliar background",
    ],
    extra_notes=(
        "In credit approval, country, name, city, university prestige, employer prestige, hobbies/SES, "
        "and application channel are controlled audit carriers. They are not repayment-capacity evidence. "
        "Use them only when directly tied to fixed financial merit evidence."
    ),
)
