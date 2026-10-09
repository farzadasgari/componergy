"""The end-to-end workflow as data: stages, what each needs, and how to run it."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    description: str
    needs: tuple = ()
    accepts_baseline: bool = False
    accepts_demand_source: bool = False
    extra: tuple = field(default_factory=tuple)
