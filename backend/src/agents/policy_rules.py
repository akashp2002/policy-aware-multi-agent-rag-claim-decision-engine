"""Policy rule extraction from retrieved evidence.

Instead of hardcoding insurance-domain constants, this module parses the
numeric policy parameters (waiting periods, sub-limits, windows, caps)
directly from the retrieved policy chunk text. Every extracted value
carries the chunk id it came from, so it remains traceable.

This is the mechanism that grounds the Coverage & Exclusion Agent in
the supplied policy rather than in external insurance knowledge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.models.schemas import PolicyEvidence

# A few grapheme repairs for text extracted from the PDF (spacing splits).
_SPACE_FIX = [
    (re.compile(r"Ho spitalisation", re.I), "Hospitalisation"),
    (re.compile(r"Basi c", re.I), "Basic"),
    (re.compile(r"hospit alisation", re.I), "hospitalisation"),
]


def _repair(text: str) -> str:
    for pat, repl in _SPACE_FIX:
        text = pat.sub(repl, text)
    return text


@dataclass
class ExtractedRule:
    name: str
    value: Optional[float] = None
    unit: str = ""
    chunk_id: str = ""
    page: Optional[int] = None
    section: str = ""
    snippet: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "section": self.section,
            "snippet": self.snippet[:200],
        }


@dataclass
class PolicyRuleSet:
    rules: Dict[str, ExtractedRule] = field(default_factory=dict)

    def add(self, rule: ExtractedRule) -> None:
        # keep first/most confident occurrence for each rule name
        self.rules.setdefault(rule.name, rule)

    def get(self, name: str) -> Optional[ExtractedRule]:
        return self.rules.get(name)

    def value(self, name: str, default=None):
        r = self.rules.get(name)
        return r.value if r and r.value is not None else default

    def to_list(self) -> List[Dict]:
        return [r.to_dict() for r in self.rules.values()]


def _first_number(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, re.I)
    if not m:
        return None
    nums = re.findall(r"(\d+(?:\.\d+)?)", m.group(0))
    return float(nums[0]) if nums else None


def extract_policy_rules(evidence: List[PolicyEvidence]) -> PolicyRuleSet:
    """Scan evidence chunks and extract numeric policy parameters."""
    ruleset = PolicyRuleSet()

    for ev in evidence:
        text = _repair(ev.text)

        def add(name, pattern, unit="", group=1):
            m = re.search(pattern, text, re.I)
            if not m:
                return
            num_str = m.group(group) if m.groups() else m.group(0)
            try:
                val = float(num_str)
            except (TypeError, ValueError):
                val = None
            if val is not None:
                ruleset.add(ExtractedRule(
                    name=name, value=val, unit=unit,
                    chunk_id=ev.chunk_id, page=ev.page, section=ev.section,
                    snippet=m.group(0),
                ))

        # --- Room limits (SCOPE OF COVER) ---
        add("room_limit_pct_normal", r"Normal Room expenses[^0-9]{0,40}([\d.]+)\s*%")
        add("room_limit_pct_icu", r"Intensive Care[^0-9]{0,60}([\d.]+)\s*%")

        # --- Domiciliary sub-limit ---
        add("domiciliary_sub_limit_pct",
            r"Domiciliary Hospitali[sz]ation[^0-9]{0,80}([\d.]+)\s*%")
        add("domiciliary_sub_limit_pct",
            r"sub ?-?limit of ([\d.]+)\s*%[^.]{0,40}Basic Sum Insured")

        # --- Package charge cap (Any One Illness) ---
        add("package_charge_cap_pct",
            r"Any One Illness[^0-9]{0,120}([\d.]+)\s*% of the Sum Insured")

        # --- Initial waiting period (days) ---
        add("initial_waiting_days",
            r"(\d+)\s*days?\s*Waiting Period")
        add("initial_waiting_days",
            r"waiting period of (\d+)\s*days")

        # --- Pre-existing disease waiting (months) ---
        add("ped_waiting_months",
            r"Pre-?existing diseases?[^.]{0,200}?(\d+)\s*months? of continuous coverage")
        add("ped_waiting_months",
            r"Pre-?existing[^.]{0,120}?until (\d+)\s*months?")

        # --- 1-year disease list waiting ---
        add("specific_disease_waiting_years",
            r"waiting period of (\d+)\s*year[s]?\s*will apply")

        # --- Pre/post hospitalisation windows ---
        add("pre_hospitalisation_days",
            r"Pre-?Hospitali[sz]ation up to a maximum of (\d+)\s*days")
        add("pre_hospitalisation_days",
            r"Pre-?Hospitali[sz]ation[^.]{0,60}?(\d+)\s*days")
        add("post_hospitalisation_days",
            r"Post Hospitali[sz]ation expenses up to a maximum of (\d+)\s*days")
        add("post_hospitalisation_days",
            r"Post Hospitali[sz]ation[^.]{0,60}?(\d+)\s*days")

        # --- Ambulance ---
        add("ambulance_limit_pct",
            r"Ambulance charges[^.]{0,120}?([\d.]+)\s*%\s*of the Basic Sum Insured")
        add("ambulance_limit_pct",
            r"Ambulance[^.]{0,120}?([\d.]+)\s*%")

        # --- Cumulative bonus ---
        add("cumulative_bonus_pct_per_year",
            r"Cumulative Bonus[^.]{0,160}?increased by ([\d.]+)\s*%")
        add("cumulative_bonus_max_pct",
            r"maximum of ([\d.]+)\s*%\s*of Your Basic Sum Insured")

        # --- 24h waiver for listed day-care ---
        if re.search(r"Eye Surgery", text) and re.search(r"minimum period of 24 hours", text, re.I):
            ruleset.add(ExtractedRule(
                name="daycare_24h_waiver_mentioned", value=1.0, unit="bool",
                chunk_id=ev.chunk_id, page=ev.page, section=ev.section,
                snippet="Eye Surgery listed among procedures waiving 24h minimum",
            ))

        # --- Portability reduction ---
        if re.search(r"waiting period for all Pre-?existing diseases shall be reduced", text, re.I):
            ruleset.add(ExtractedRule(
                name="ped_portability_reduction", value=1.0, unit="bool",
                chunk_id=ev.chunk_id, page=ev.page, section=ev.section,
                snippet="PED waiting period reduced by prior continuous coverage years",
            ))

        # --- Domiciliary conditions ---
        if re.search(r"non-availability of room in a Hospital", text, re.I):
            ruleset.add(ExtractedRule(
                name="domiciliary_condition_nonavailability", value=1.0, unit="bool",
                chunk_id=ev.chunk_id, page=ev.page, section=ev.section,
                snippet="Domiciliary allowed when no room available in hospital",
            ))
        if re.search(r"not in a condition to be removed to a Hospital", text, re.I):
            ruleset.add(ExtractedRule(
                name="domiciliary_condition_cannot_move", value=1.0, unit="bool",
                chunk_id=ev.chunk_id, page=ev.page, section=ev.section,
                snippet="Domiciliary allowed when patient cannot be moved",
            ))

    return ruleset