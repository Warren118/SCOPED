# fair_skills/fair_skills.py
# -*- coding: utf-8 -*-
"""
Backward-compatible entrypoint.

run_suite.py can keep importing:
    from fair_skills import build_skill_pressure, fair_skill_metadata

The implementation is now split into:
- core.py: generic data structures
- templates.py: domain-general Fair Skill templates and profiles
- bindings/hiring.py: hiring scenario adapter/binding
- renderer.py: profile + binding -> prompt
- runtime.py: API used by run_suite
"""

from .runtime import (
    build_skill_pressure,
    fair_skill_metadata,
    build_fair_skill_instruction,
)
from .renderer import get_profile
