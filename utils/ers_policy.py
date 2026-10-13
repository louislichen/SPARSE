from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


class ERSPolicyError(ValueError):
    """Raised when an ERS calibration policy is ambiguous or invalid."""


@dataclass(frozen=True)
class ERSPolicyRule:
    name: str
    layers: Tuple[int, ...]
    positions: Tuple[int, ...]
    max_replace: Optional[int]
    margin_min: Optional[float]
    margin_max: Optional[float]
    candidate_mode: str
    top_m: Optional[int]
    packet_critical_only: bool
    raw: str


@dataclass(frozen=True)
class ERSPolicy:
    raw: str
    num_layers: int
    top_k: int
    rules: Tuple[ERSPolicyRule, ...]


def normalize_candidate_mode(mode: str) -> str:
    m = str(mode or "same_node").strip().lower()
    if m in ("same-node", "same_node", "local_node", "local-node"):
        return "same_node"
    if m in ("owner-local", "owner_local", "local_rank", "local-rank"):
        return "owner_local"
    if m in ("any", "any_non_topk", "any-non-topk"):
        return "any_non_topk"
    if m in ("remaining-node", "remaining_node", "remaining"):
        return "remaining_node"
    raise ERSPolicyError(f"unsupported ERS candidate_mode={mode!r}")


def _parse_bool(raw: str, *, default: bool = False) -> bool:
    text = str(raw or "").strip().lower()
    if not text:
        return bool(default)
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ERSPolicyError(f"invalid boolean value {raw!r}")


def parse_int_set(raw: str, *, num_layers: Optional[int] = None, allow_none: bool = True) -> Optional[Set[int]]:
    text = str(raw or "").strip().lower()
    if not text:
        return None if allow_none else set()
    if text in ("all", "*"):
        if num_layers is None:
            return None
        return set(range(int(num_layers)))
    if text in ("none", "off", "disabled"):
        return set()

    aliases = {
        "early": "0-5",
        "middle": "6-17",
        "mid": "6-17",
        "late": "18-23",
    }
    out: Set[int] = set()
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        part = aliases.get(part, part)
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            if not lo_s.strip() or not hi_s.strip():
                raise ERSPolicyError(f"invalid integer range {part!r}")
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise ERSPolicyError(f"invalid descending integer range {part!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if num_layers is not None:
        bad = sorted(x for x in out if x < 0 or x >= int(num_layers))
        if bad:
            raise ERSPolicyError(f"layer ids out of range 0..{int(num_layers) - 1}: {bad}")
    return out


def parse_margin(raw: str) -> Tuple[Optional[float], Optional[float]]:
    text = str(raw or "").strip().lower()
    if not text or text in ("none", "off", "disabled", "no_margin"):
        return None, None
    if text.startswith("<="):
        return None, float(text[2:].strip())
    if text.startswith("<"):
        return None, float(text[1:].strip())
    if text.startswith(">="):
        return float(text[2:].strip()), None
    if ".." in text:
        lo_s, hi_s = text.split("..", 1)
        lo = None if not lo_s.strip() else float(lo_s)
        hi = None if not hi_s.strip() else float(hi_s)
    elif "-" in text:
        lo_s, hi_s = text.split("-", 1)
        lo = None if not lo_s.strip() else float(lo_s)
        hi = None if not hi_s.strip() else float(hi_s)
    else:
        lo = None
        hi = float(text)
    if lo is not None and hi is not None and float(lo) > float(hi):
        raise ERSPolicyError(f"invalid margin range {raw!r}: low > high")
    return lo, hi


def _split_rule_fields(body: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    # The formal examples use comma-separated fields, while position lists also
    # use commas (pos=2,3). Split only on separators followed by another known
    # key/operator; ``|`` remains the least ambiguous supported separator.
    key_names = (
        "layers", "layer", "l", "positions", "position", "pos",
        "max_replace", "replace", "r", "margin", "m", "margin_min",
        "margin_max", "candidate_mode", "candidate", "topm", "top_m",
        "packet_critical_only", "packet_critical",
    )
    key_pat = "|".join(re.escape(k) for k in key_names)
    text = str(body or "")
    chunks: List[str] = []
    for pipe_part in text.split("|"):
        chunks.extend(re.split(rf",(?=\s*(?:{key_pat})\s*(?:=|<=|>=))", pipe_part, flags=re.IGNORECASE))
    for part in chunks:
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ERSPolicyError(f"invalid policy field {part!r}; expected key=value")
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        # Allow compact comparison syntax such as margin<=0.10 and margin>=0.05.
        if key == "margin<":
            fields["margin"] = "<=" + value
            continue
        if key == "margin>":
            fields["margin"] = ">=" + value
            continue
        fields[key] = value
    return fields


def parse_policy(
    raw: str,
    *,
    num_layers: int = 24,
    top_k: int = 4,
    default_candidate_mode: str = "same_node",
    default_top_m: Optional[int] = None,
) -> ERSPolicy:
    text = str(raw or "").strip()
    if not text:
        return ERSPolicy(raw="", num_layers=int(num_layers), top_k=int(top_k), rules=())

    rules: List[ERSPolicyRule] = []
    covered: Dict[int, str] = {}
    for idx, raw_rule in enumerate(text.split(";")):
        raw_rule = raw_rule.strip()
        if not raw_rule:
            continue
        name = f"segment_{idx}"
        body = raw_rule
        if ":" in raw_rule:
            maybe_name, maybe_body = raw_rule.split(":", 1)
            if "=" in maybe_body:
                name = maybe_name.strip() or name
                body = maybe_body.strip()
        fields = _split_rule_fields(body)

        layers_raw = fields.get("layers", fields.get("layer", fields.get("l", "")))
        if not layers_raw:
            raise ERSPolicyError(f"policy segment {name!r} is missing layers=...")
        layers = parse_int_set(layers_raw, num_layers=int(num_layers), allow_none=False) or set()
        if not layers:
            continue

        pos_raw = fields.get("positions", fields.get("position", fields.get("pos", "")))
        if not pos_raw:
            raise ERSPolicyError(f"policy segment {name!r} is missing pos=...")
        positions = parse_int_set(pos_raw, num_layers=None, allow_none=False) or set()
        bad_pos = sorted(p for p in positions if p < 0 or p >= int(top_k))
        if bad_pos:
            raise ERSPolicyError(f"policy segment {name!r} has invalid top-k positions {bad_pos}; top_k={top_k}")

        max_replace: Optional[int] = None
        if "max_replace" in fields:
            max_replace = int(fields["max_replace"])
        elif "replace" in fields:
            max_replace = int(fields["replace"])
        if max_replace is not None and max_replace < 0:
            raise ERSPolicyError(f"policy segment {name!r} has negative max_replace={max_replace}")

        margin_min, margin_max = parse_margin(fields.get("margin", fields.get("m", "")))
        if "margin_min" in fields:
            margin_min = float(fields["margin_min"])
        if "margin_max" in fields:
            margin_max = float(fields["margin_max"])
        if margin_min is not None and margin_max is not None and margin_min > margin_max:
            raise ERSPolicyError(f"policy segment {name!r} has margin_min > margin_max")

        candidate_mode = normalize_candidate_mode(
            fields.get("candidate_mode", fields.get("candidate", default_candidate_mode))
        )
        top_m = default_top_m
        if "topm" in fields:
            top_m = int(fields["topm"])
        elif "top_m" in fields:
            top_m = int(fields["top_m"])
        if top_m is not None and top_m <= 0:
            raise ERSPolicyError(f"policy segment {name!r} has invalid topM={top_m}")

        packet_critical_only = _parse_bool(
            fields.get("packet_critical_only", fields.get("packet_critical", "")),
            default=False,
        )

        overlap = sorted(layer for layer in layers if int(layer) in covered)
        if overlap:
            prev = ", ".join(f"L{x}:{covered[int(x)]}" for x in overlap[:8])
            raise ERSPolicyError(f"overlapping ERS policy layers in segment {name!r}: {prev}")
        for layer in layers:
            covered[int(layer)] = name

        rules.append(
            ERSPolicyRule(
                name=name,
                layers=tuple(sorted(int(x) for x in layers)),
                positions=tuple(sorted(int(x) for x in positions)),
                max_replace=max_replace,
                margin_min=margin_min,
                margin_max=margin_max,
                candidate_mode=candidate_mode,
                top_m=top_m,
                packet_critical_only=packet_critical_only,
                raw=raw_rule,
            )
        )
    return ERSPolicy(raw=text, num_layers=int(num_layers), top_k=int(top_k), rules=tuple(rules))


def find_rule_for_layer(policy: ERSPolicy, layer: int) -> Optional[ERSPolicyRule]:
    lid = int(layer)
    for rule in policy.rules:
        if lid in set(rule.layers):
            return rule
    return None


def policy_to_layer_rows(policy: ERSPolicy) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for layer in range(int(policy.num_layers)):
        rule = find_rule_for_layer(policy, layer)
        if rule is None:
            rows.append({
                "layer": layer,
                "segment_name": "",
                "positions": "",
                "max_replace": "",
                "margin_low": "",
                "margin_high": "",
                "candidate_mode": "",
                "topM": "",
                "packet_critical_only": "",
            })
            continue
        rows.append({
            "layer": layer,
            "segment_name": rule.name,
            "positions": ",".join(str(x) for x in rule.positions),
            "max_replace": "" if rule.max_replace is None else int(rule.max_replace),
            "margin_low": "" if rule.margin_min is None else float(rule.margin_min),
            "margin_high": "" if rule.margin_max is None else float(rule.margin_max),
            "candidate_mode": rule.candidate_mode,
            "topM": "" if rule.top_m is None else int(rule.top_m),
            "packet_critical_only": int(bool(rule.packet_critical_only)),
        })
    return rows


def policy_to_json_dict(policy: ERSPolicy) -> Dict[str, object]:
    return {
        "raw": policy.raw,
        "num_layers": policy.num_layers,
        "top_k": policy.top_k,
        "rules": [asdict(rule) for rule in policy.rules],
    }


def write_policy_outputs(policy: ERSPolicy, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parsed_policy.json").write_text(
        json.dumps(policy_to_json_dict(policy), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = policy_to_layer_rows(policy)
    with (out / "per_layer_policy.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
