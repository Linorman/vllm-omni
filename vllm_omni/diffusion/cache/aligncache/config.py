# SPDX-License-Identifier: Apache-2.0
"""Strict, immutable schema for the uncalibrated wan21_reference_v1 profile."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Phases:
    early: float
    mid: float
    late: float

    def __getitem__(self, phase: str):
        return getattr(self, phase)


@dataclass(frozen=True)
class PhaseConfig:
    early_end: float = 0.15
    late_start: float = 0.86
    scales: Phases = Phases(1.15, 1.0, 0.8)


@dataclass(frozen=True)
class FeedbackConfig:
    age_caps: Phases = Phases(5, 4, 3)
    scale_init: Phases = Phases(1.0, 1.0, 1.0)
    scale_low: Phases = Phases(0.95, 0.90, 0.85)
    scale_high: Phases = Phases(1.03, 1.06, 1.0)
    ratio_ema: float = 0.90
    ratio_low: float = 0.85
    ratio_high: float = 1.25
    eps: float = 1e-6


@dataclass(frozen=True)
class PredictorConfig:
    enabled: bool = True
    second_order: bool = True
    online_scale: bool = True
    stride: int = 64
    ratio_ema: float = 0.8
    direction_ema: float = 0.8
    beta: Phases = Phases(0.0, 0.35, 0.15)
    ratio_tolerance: Phases = Phases(0.0, 0.30, 0.18)
    direction_min: Phases = Phases(1.0, 0.80, 0.88)
    scale_ema: float = 0.85
    scale_low: float = 0.85
    scale_high: float = 1.18


@dataclass(frozen=True)
class AnchorConfig:
    enabled: bool = True
    early_exact: bool = True
    age_caps: Phases = Phases(0, 1, 2)
    cond_cap: Phases = Phases(0.0, 0.005, 0.020)
    disagreement_cap: Phases = Phases(0.0, 0.020, 0.060)
    alpha_init: Phases = Phases(1.279282, 1.018750, 0.972800)
    alpha_ema: float = 0.85
    blend: Phases = Phases(0.0, 0.35, 0.20)


@dataclass(frozen=True)
class AlignCacheConfig:
    schema_version: int = 1
    policy: str = "wan21_feedback_v1"
    profile: str = "wan21_reference_v1"
    mode: str = "accelerate"
    base_threshold: float = 0.065
    warmup_steps: int = 12
    tail_steps: int = 1
    max_cache_bytes: int = 1073741824
    diagnostics: str = "summary"
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def parse(cls, value: Any = None) -> AlignCacheConfig:
        namespace = normalize_namespace(value)
        defaults = cls()
        profile = namespace.get("profile", "wan21_reference_v1")
        if profile == "wan22_t2v_reference_v1":
            defaults = replace(
                defaults, policy="wan22_feedback_v1", profile=profile, base_threshold=0.078, warmup_steps=9
            )
        elif profile == "wan21_14b_reference_v1":
            defaults = replace(defaults, profile=profile)
        config = _merge(defaults, namespace, "align_cache")
        _validate(config)
        return config


def normalize_namespace(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("cache_config: expected a valid JSON object") from exc
        if not isinstance(value, dict):
            raise ValueError("cache_config: JSON must be an object")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("cache_config: field names must be strings")
        unknown = set(value) - {"align_cache"}
        if unknown:
            raise ValueError(f"cache_config: align_cache cannot use other backend/unknown fields: {sorted(unknown)}")
        namespace = value.get("align_cache", {})
    else:
        # Shared config defaults belong to other backends and are not explicit overrides.
        if not is_dataclass(value):
            raise ValueError("cache_config: expected None, dict, JSON object or DiffusionCacheConfig")
        from vllm_omni.diffusion.data import DiffusionCacheConfig

        if not isinstance(value, DiffusionCacheConfig):
            raise ValueError("cache_config: expected None, dict, JSON object or DiffusionCacheConfig")
        defaults = DiffusionCacheConfig()
        explicit = getattr(value, "_explicit_fields", ())
        foreign = set(explicit) - {"align_cache"}
        foreign.update(
            k
            for k, v in vars(value).items()
            if not k.startswith("_") and k != "align_cache" and v != getattr(defaults, k)
        )
        if foreign or value._extra_params:
            raise ValueError(f"cache_config: incompatible/unknown fields {sorted(foreign | set(value._extra_params))}")
        namespace = value.align_cache
    if not isinstance(namespace, dict):
        raise ValueError("cache_config.align_cache: expected object")
    return namespace


def normalize_cache_config(value: Any) -> dict:
    """Preserve the explicit namespace after checking all entry-point inputs."""
    namespace = normalize_namespace(value)
    AlignCacheConfig.parse({"align_cache": namespace})
    return {"align_cache": namespace}


def _merge(default, overrides, path):
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: expected object")
    if any(not isinstance(key, str) for key in overrides):
        raise ValueError(f"{path}: field names must be strings")
    names = {f.name for f in fields(default)}
    if set(overrides) - names:
        raise ValueError(f"{path}: unknown fields {sorted(set(overrides) - names)}; legal fields: {sorted(names)}")
    changes = {}
    for key, value in overrides.items():
        current = getattr(default, key)
        label = f"{path}.{key}"
        if is_dataclass(current):
            changes[key] = _merge(current, value, label)
        elif isinstance(current, bool):
            if type(value) is not bool:
                raise ValueError(f"{label}: expected bool")
            changes[key] = value
        elif isinstance(current, (int, float)):
            valid = type(value) is int if type(current) is int else type(value) in (int, float)
            try:
                finite = math.isfinite(value) if valid else False
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{label}: expected finite {type(current).__name__}, not bool")
            changes[key] = value if type(current) is int else float(value)
        elif not isinstance(value, str):
            raise ValueError(f"{label}: expected string")
        else:
            changes[key] = value
    return replace(default, **changes)


def _validate(c):
    for key, legal in {
        "schema_version": (1,),
        "policy": ("wan21_feedback_v1", "wan22_feedback_v1"),
        "profile": ("wan21_reference_v1", "wan21_14b_reference_v1", "wan22_t2v_reference_v1"),
        "mode": ("passthrough", "force_exact", "trace", "accelerate"),
        "diagnostics": ("off", "summary", "trace"),
    }.items():
        if getattr(c, key) not in legal:
            raise ValueError(f"align_cache.{key}: legal values {legal}")

    def check(ok, path, rule):
        if not ok:
            raise ValueError(f"align_cache.{path}: {rule}")

    check(
        (c.policy == "wan22_feedback_v1") == (c.profile == "wan22_t2v_reference_v1"),
        "profile",
        "Wan2.2 policy requires wan22_t2v_reference_v1; Wan2.1 requires a Wan2.1 profile",
    )
    check(c.base_threshold >= 0, "base_threshold", "must be >= 0")
    for name in ("warmup_steps", "tail_steps"):
        check(getattr(c, name) >= 0, name, "must be >= 0")
    check(c.max_cache_bytes > 0, "max_cache_bytes", "must be > 0")
    check(0 <= c.phase.early_end < c.phase.late_start <= 1, "phase", "require 0 <= early_end < late_start <= 1")
    check(c.predictor.stride == 64, "predictor.stride", "wan21_feedback_v1 requires 64")
    for group in ("feedback", "predictor", "anchor"):
        obj = getattr(c, group)
        for f in fields(obj):
            value = getattr(obj, f.name)
            if f.name.endswith("ema"):
                check(0 <= value < 1, f"{group}.{f.name}", "require 0 <= decay < 1")
        if hasattr(obj, "scale_low") and not is_dataclass(obj.scale_low):
            check(0 < obj.scale_low <= obj.scale_high, f"{group}.scale_low", "require 0 < low <= high")
    check(0 < c.feedback.ratio_low <= c.feedback.ratio_high, "feedback.ratio_low", "require 0 < low <= high")
    check(c.feedback.eps > 0, "feedback.eps", "must be > 0")
    for p in ("early", "mid", "late"):
        check(c.phase.scales[p] >= 0, f"phase.scales.{p}", "must be >= 0")
        check(
            0 < c.feedback.scale_low[p] <= c.feedback.scale_init[p] <= c.feedback.scale_high[p],
            f"feedback.scale_init.{p}",
            "require 0 < low <= init <= high",
        )
        for group in ("feedback", "anchor"):
            check(getattr(c, group).age_caps[p] >= 0, f"{group}.age_caps.{p}", "must be >= 0")
        for key in ("beta", "ratio_tolerance"):
            check(getattr(c.predictor, key)[p] >= 0, f"predictor.{key}.{p}", "must be >= 0")
        check(-1 <= c.predictor.direction_min[p] <= 1, f"predictor.direction_min.{p}", "require [-1, 1]")
        for key in ("cond_cap", "disagreement_cap"):
            check(getattr(c.anchor, key)[p] >= 0, f"anchor.{key}.{p}", "must be >= 0")
        check(0 <= c.anchor.blend[p] <= 1, f"anchor.blend.{p}", "require [0, 1]")
        check(-1.5 <= c.anchor.alpha_init[p] <= 1.5, f"anchor.alpha_init.{p}", "require [-1.5, 1.5]")
