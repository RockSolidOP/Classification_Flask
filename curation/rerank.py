from __future__ import annotations

import re
from typing import Dict, List, Optional


def _canonical(label: str, aliases: Dict[str, str]) -> str:
    return aliases.get(label, label)


def _base_and_page(label: str) -> (str, Optional[int]):
    m = re.match(r"^(.+)_P(\d+)$", label)
    if m:
        try:
            return m.group(1), int(m.group(2))
        except Exception:
            return m.group(1), None
    return label, None


def _page_token_from_raw(raw_label: str) -> Optional[int]:
    if not raw_label:
        return None
    m = re.search(r"\bP(\d+)\b", raw_label)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def rerank_suggestions(
    results: List[Dict],
    page_entry: Optional[Dict],
    prev_entry: Optional[Dict],
    aliases: Dict[str, str],
    topk: int = 5,
) -> List[Dict]:
    """Canonicalize, dedupe and re-rank suggestions using light priors.

    Boosts:
      - +0.03 if candidate label equals page auto_label (after aliases)
      - +0.02 if candidate page number (_P#) matches a token in raw_label/label
      - +0.02 if candidate base_label equals previous page's base_label (continuity)
    """
    # Canonicalize + dedupe by canonical label (keep best raw score)
    agg: Dict[str, Dict] = {}
    for r in results or []:
        can = _canonical(str(r.get("label", "")), aliases)
        try:
            raw_score = float(r.get("score", 0.0))
        except Exception:
            raw_score = 0.0
        kept = agg.get(can)
        if kept is None or raw_score > float(kept.get("raw_score", 0.0)):
            nr = dict(r)
            nr["label"] = can
            nr["raw_score"] = raw_score
            agg[can] = nr

    merged = list(agg.values())

    # Compute priors from page/prev context
    auto_can = None
    page_num_hint: Optional[int] = None
    prev_base: Optional[str] = None
    expected_base: Optional[str] = None

    if page_entry:
        al = page_entry.get("auto_label") or page_entry.get("label") or ""
        auto_can = _canonical(str(al), aliases)
        # P# from raw_label first, then from current label
        page_num_hint = _page_token_from_raw(str(page_entry.get("raw_label", "")))
        _cur_base, _cur_p = _base_and_page(auto_can)
        expected_base = _cur_base
        if page_num_hint is None:
            page_num_hint = _cur_p

    if prev_entry:
        prev_lbl = prev_entry.get("label") or prev_entry.get("auto_label") or ""
        prev_can = _canonical(str(prev_lbl), aliases)
        prev_base, _ = _base_and_page(prev_can)

    # Apply small boosts
    for m in merged:
        score = float(m.get("raw_score", 0.0))
        can = str(m.get("label", ""))
        base, pnum = _base_and_page(can)
        if auto_can and can == auto_can:
            score += 0.03
        # Only boost page-number match if the base label also matches the expected base
        if (
            page_num_hint is not None
            and pnum is not None
            and pnum == page_num_hint
            and expected_base is not None
            and base == expected_base
        ):
            score += 0.02
        if prev_base and base == prev_base:
            score += 0.02
        m["score"] = score

    # Sort and cut
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    merged = merged[: topk or 5]
    for i, m in enumerate(merged, start=1):
        m["rank"] = i
    return merged
