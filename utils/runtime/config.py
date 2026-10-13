from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_FALSE_STRINGS = ("", "0", "false", "no", "off")
_TRUE_STRINGS = ("1", "true", "yes", "on")


def _normalize_env_name(name: str) -> str:
    return str(name).strip()


def env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.environ.get(_normalize_env_name(name), default)).strip().lower()
    return raw not in _FALSE_STRINGS


def env_value(name: str, default: str = "") -> str:
    return str(os.environ.get(_normalize_env_name(name), default))


def env_value_with_alias(name: str, legacy_name: str = "", default: str = "") -> str:
    """Read a canonical SPARSE_* env var with one deprecated fallback.

    The canonical name wins if both are set. This is intentionally quiet in the
    runtime hot path; summaries can call ``ers_config_from_env`` to expose the
    deprecated aliases that were used.
    """
    name = _normalize_env_name(name)
    legacy_name = _normalize_env_name(legacy_name)
    if name in os.environ:
        return str(os.environ.get(name, default))
    if legacy_name and legacy_name in os.environ:
        return str(os.environ.get(legacy_name, default))
    return str(default)


def env_flag_with_alias(name: str, legacy_name: str = "", default: str = "0") -> bool:
    raw = env_value_with_alias(name, legacy_name, default).strip().lower()
    return raw not in _FALSE_STRINGS


def _parse_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _parse_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class EnvAlias:
    canonical: str
    legacy: str
    default: str
    description: str = ""


ERS_ENV_ALIASES: Tuple[EnvAlias, ...] = (
    EnvAlias("SPARSE_ERS_ENABLE", "SPARSE_NODEPATCH_ENABLE", "0", "Enable ERS runtime skeleton/legacy NodePatch alias."),
    EnvAlias("SPARSE_ERS_MODE", "SPARSE_NODEPATCH_MODE", "legacy_nodepatch", "ERS runtime mode; default is legacy adapter when enabled."),
    EnvAlias("SPARSE_ERS_ACTION", "SPARSE_NODEPATCH_ACTION", "replace", "ERS runtime action: replace or drop/prune low-value routed experts."),
    EnvAlias("SPARSE_ERS_PHASE", "SPARSE_NODEPATCH_PHASE", "decode", "prefill/decode/all phase gate."),
    EnvAlias("SPARSE_ERS_TOPM", "SPARSE_NODEPATCH_TOPM", "8", "Router candidate depth."),
    EnvAlias("SPARSE_ERS_MAX_REPLACE", "SPARSE_NODEPATCH_MAX_REPLACE", "1", "Max expert replacements per token/packet in legacy adapter."),
    EnvAlias("SPARSE_ERS_MARGIN", "SPARSE_NODEPATCH_MARGIN", "0.05", "Router score margin threshold."),
    EnvAlias("SPARSE_ERS_MARGIN_KIND", "SPARSE_NODEPATCH_MARGIN_KIND", "logit", "logit/prob margin semantics."),
    EnvAlias("SPARSE_ERS_KEEP_TOP1", "SPARSE_NODEPATCH_KEEP_TOP1", "1", "Protect top-1 router choice."),
    EnvAlias("SPARSE_ERS_ONLY_CROSS_NODE", "SPARSE_NODEPATCH_ONLY_CROSS_NODE", "1", "Only consider cross-node choices."),
    EnvAlias("SPARSE_ERS_DISABLE_MARGIN", "SPARSE_NODEPATCH_DISABLE_MARGIN", "0", "Disable score margin check."),
    EnvAlias("SPARSE_ERS_FASTPATH", "SPARSE_NODEPATCH_FASTPATH", "0", "Legacy runtime fastpath alias."),
    EnvAlias("SPARSE_ERS_COLLECT_HIST", "SPARSE_NODEPATCH_COLLECT_HIST", "0", "Collect repair histograms."),
    EnvAlias("SPARSE_ERS_EVENT_AWARE", "SPARSE_NODEPATCH_EVENT_AWARE", "0", "Use event-aware legacy adapter."),
    EnvAlias("SPARSE_ERS_EVENT_AGGRESSIVE", "SPARSE_NODEPATCH_EVENT_AGGRESSIVE", "0", "Use aggressive legacy adapter."),
    EnvAlias("SPARSE_ERS_EVENT_ONLY_EMPTY_PACKET", "SPARSE_NODEPATCH_EVENT_ONLY_EMPTY_PACKET", "1", "Only count packet-emptying repair."),
    EnvAlias("SPARSE_ERS_EVENT_REQUIRE_LOCAL_CANDIDATE", "SPARSE_NODEPATCH_EVENT_REQUIRE_LOCAL_CANDIDATE", "1", "Require local replacement candidate."),
    EnvAlias("SPARSE_ERS_EVENT_PACKET_MAX_ITEMS", "SPARSE_NODEPATCH_EVENT_PACKET_MAX_ITEMS", "4", "Packet item cap for legacy runtime search."),
    EnvAlias("SPARSE_ERS_PACKET_EMPTYING_FIRST", "SPARSE_NODEPATCH_PACKET_EMPTYING_FIRST", "1", "Prioritize packet-emptying repairs."),
    EnvAlias("SPARSE_ERS_CACHE_AWARE", "SPARSE_NODEPATCH_CACHE_AWARE", "0", "Use cache-aware filter."),
    EnvAlias("SPARSE_ERS_CACHE_REQUIRE_OWNER_LOCAL", "SPARSE_NODEPATCH_CACHE_REQUIRE_OWNER_LOCAL", "1", "Require owner-local added expert under cache-aware mode."),
    EnvAlias("SPARSE_ERS_REQUIRE_ADDED_RESIDENT", "SPARSE_NODEPATCH_REQUIRE_ADDED_RESIDENT", "1", "Require added expert already resident."),
    EnvAlias("SPARSE_ERS_DROP_RENORMALIZE", "SPARSE_NODEPATCH_DROP_RENORMALIZE", "1", "Renormalize remaining top-k weights after ERS drop."),
    EnvAlias("SPARSE_ERS_DROP_REQUIRE_EVENT_REDUCTION", "SPARSE_NODEPATCH_DROP_REQUIRE_EVENT_REDUCTION", "1", "Only drop a routed expert if token locality proxy improves."),
    EnvAlias("SPARSE_ERS_DROP_MAX_WEIGHT", "SPARSE_NODEPATCH_DROP_MAX_WEIGHT", "", "Optional max router probability for a dropped expert."),
)


@dataclass(frozen=True)
class ERSRuntimeConfig:
    enabled: bool = False
    mode: str = "legacy_nodepatch"
    action: str = "replace"
    phase: str = "decode"
    top_m: int = 8
    max_replace: int = 1
    margin: float = 0.05
    margin_kind: str = "logit"
    keep_top1: bool = True
    only_cross_node: bool = True
    disable_margin: bool = False
    fastpath: bool = False
    collect_hist: bool = False
    event_aware: bool = False
    event_aggressive: bool = False
    event_only_empty_packet: bool = True
    event_require_local_candidate: bool = True
    event_packet_max_items: int = 4
    packet_emptying_first: bool = True
    cache_aware: bool = False
    cache_require_owner_local: bool = True
    require_added_resident: bool = True
    drop_renormalize: bool = True
    drop_require_event_reduction: bool = True
    drop_max_weight: Optional[float] = None
    deprecated_aliases_used: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def legacy_alias_used(self) -> bool:
        return bool(self.deprecated_aliases_used)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "ers_enabled": int(self.enabled),
            "ers_mode": self.mode,
            "ers_action": self.action,
            "ers_phase": self.phase,
            "ers_topm": int(self.top_m),
            "ers_max_replace": int(self.max_replace),
            "ers_margin": float(self.margin),
            "ers_margin_kind": self.margin_kind,
            "ers_keep_top1": int(self.keep_top1),
            "ers_only_cross_node": int(self.only_cross_node),
            "ers_disable_margin": int(self.disable_margin),
            "ers_fastpath": int(self.fastpath),
            "ers_collect_hist": int(self.collect_hist),
            "ers_event_aware": int(self.event_aware),
            "ers_event_aggressive": int(self.event_aggressive),
            "ers_event_only_empty_packet": int(self.event_only_empty_packet),
            "ers_event_require_local_candidate": int(self.event_require_local_candidate),
            "ers_event_packet_max_items": int(self.event_packet_max_items),
            "ers_packet_emptying_first": int(self.packet_emptying_first),
            "ers_cache_aware": int(self.cache_aware),
            "ers_cache_require_owner_local": int(self.cache_require_owner_local),
            "ers_require_added_resident": int(self.require_added_resident),
            "ers_drop_renormalize": int(self.drop_renormalize),
            "ers_drop_require_event_reduction": int(self.drop_require_event_reduction),
            "ers_drop_max_weight": self.drop_max_weight,
            "ers_legacy_nodepatch_alias_used": int(self.legacy_alias_used),
            "ers_deprecated_aliases_used": list(self.deprecated_aliases_used),
            "ers_runtime_semantics": "ERS is default-off; SPARSE_NODEPATCH_* is a deprecated legacy adapter alias.",
        }


def _read_alias(alias: EnvAlias) -> Tuple[str, Optional[str]]:
    if alias.canonical in os.environ:
        return str(os.environ.get(alias.canonical, alias.default)), None
    if alias.legacy and alias.legacy in os.environ:
        return str(os.environ.get(alias.legacy, alias.default)), alias.legacy
    return alias.default, None


def _flag(raw: str) -> bool:
    return str(raw).strip().lower() not in _FALSE_STRINGS


def ers_config_from_env() -> ERSRuntimeConfig:
    vals: Dict[str, str] = {}
    used: List[str] = []
    for alias in ERS_ENV_ALIASES:
        value, legacy = _read_alias(alias)
        vals[alias.canonical] = value
        if legacy:
            used.append(legacy)
    return ERSRuntimeConfig(
        enabled=_flag(vals["SPARSE_ERS_ENABLE"]),
        mode=str(vals["SPARSE_ERS_MODE"] or "legacy_nodepatch").strip().lower(),
        action=str(vals["SPARSE_ERS_ACTION"] or "replace").strip().lower(),
        phase=str(vals["SPARSE_ERS_PHASE"] or "decode").strip().lower(),
        top_m=_parse_int(vals["SPARSE_ERS_TOPM"], 8),
        max_replace=_parse_int(vals["SPARSE_ERS_MAX_REPLACE"], 1),
        margin=_parse_float(vals["SPARSE_ERS_MARGIN"], 0.05),
        margin_kind=str(vals["SPARSE_ERS_MARGIN_KIND"] or "logit").strip().lower(),
        keep_top1=_flag(vals["SPARSE_ERS_KEEP_TOP1"]),
        only_cross_node=_flag(vals["SPARSE_ERS_ONLY_CROSS_NODE"]),
        disable_margin=_flag(vals["SPARSE_ERS_DISABLE_MARGIN"]),
        fastpath=_flag(vals["SPARSE_ERS_FASTPATH"]),
        collect_hist=_flag(vals["SPARSE_ERS_COLLECT_HIST"]),
        event_aware=_flag(vals["SPARSE_ERS_EVENT_AWARE"]),
        event_aggressive=_flag(vals["SPARSE_ERS_EVENT_AGGRESSIVE"]),
        event_only_empty_packet=_flag(vals["SPARSE_ERS_EVENT_ONLY_EMPTY_PACKET"]),
        event_require_local_candidate=_flag(vals["SPARSE_ERS_EVENT_REQUIRE_LOCAL_CANDIDATE"]),
        event_packet_max_items=_parse_int(vals["SPARSE_ERS_EVENT_PACKET_MAX_ITEMS"], 4),
        packet_emptying_first=_flag(vals["SPARSE_ERS_PACKET_EMPTYING_FIRST"]),
        cache_aware=_flag(vals["SPARSE_ERS_CACHE_AWARE"]),
        cache_require_owner_local=_flag(vals["SPARSE_ERS_CACHE_REQUIRE_OWNER_LOCAL"]),
        require_added_resident=_flag(vals["SPARSE_ERS_REQUIRE_ADDED_RESIDENT"]),
        drop_renormalize=_flag(vals["SPARSE_ERS_DROP_RENORMALIZE"]),
        drop_require_event_reduction=_flag(vals["SPARSE_ERS_DROP_REQUIRE_EVENT_REDUCTION"]),
        drop_max_weight=(
            None
            if str(vals["SPARSE_ERS_DROP_MAX_WEIGHT"] or "").strip() == ""
            else _parse_float(vals["SPARSE_ERS_DROP_MAX_WEIGHT"], 0.0)
        ),
        deprecated_aliases_used=tuple(sorted(set(used))),
    )


@dataclass(frozen=True)
class SPAREMainlineConfig:
    code_env_prefix: str = "SPARSE_*"
    paper_system_name: str = "SPARE"
    default_runtime_cases: Tuple[str, ...] = ("rr_normal", "rs_prefetch", "rr_node_grouped", "rr_node_grouped_rs")
    default_enabled_modules: Tuple[str, ...] = ("RS", "NGD")
    default_gate_required_modules: Tuple[str, ...] = ("SCP", "ERS")
    metrics_semantics: str = (
        "Runtime wall metrics, logical runtime counters, weak-interconnect cost-model metrics, "
        "physical measurements, and offline proxies must be reported separately."
    )


DEFAULT_SPARE_MAINLINE = SPAREMainlineConfig()
