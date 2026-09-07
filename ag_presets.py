"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem
OR-Demo-Portfolio, siehe db_presets.py in dbscan-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import ag_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _linkage_caster(v):
    return v if v in C.LINKAGES else C.DEFAULT_LINKAGE


def _shape_caster(v):
    return v if v in C.SHAPES else C.DEFAULT_SHAPE


SETTING_SPECS = {
    "n_points_slider": SettingSpec("n", int, C.DEFAULT_N_POINTS, C.N_POINTS_MIN, C.N_POINTS_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "spread_slider": SettingSpec("spread", float, C.DEFAULT_SPREAD, C.SPREAD_MIN, C.SPREAD_MAX),
    "size_imbalance_slider": SettingSpec(
        "simb", float, C.DEFAULT_SIZE_IMBALANCE, C.SIZE_IMBALANCE_MIN, C.SIZE_IMBALANCE_MAX
    ),
    "bridge_strength_slider": SettingSpec(
        "bridge", float, C.DEFAULT_BRIDGE_STRENGTH, C.BRIDGE_STRENGTH_MIN, C.BRIDGE_STRENGTH_MAX
    ),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
    "shape_radio": SettingSpec("shape", _shape_caster, C.DEFAULT_SHAPE),
    "linkage_radio": SettingSpec("linkage", _linkage_caster, C.DEFAULT_LINKAGE),
    "target_k_slider": SettingSpec("tk", int, C.DEFAULT_TARGET_K, C.TARGET_K_MIN, C.TARGET_K_MAX),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(n_points, k, spread, size_imbalance, bridge_strength, seed, shape, linkage, target_k):
    try:
        st.query_params["n"] = str(int(n_points))
        st.query_params["k"] = str(int(k))
        st.query_params["spread"] = str(spread)
        st.query_params["simb"] = str(size_imbalance)
        st.query_params["bridge"] = str(bridge_strength)
        st.query_params["seed"] = str(int(seed))
        st.query_params["shape"] = shape
        st.query_params["linkage"] = linkage
        st.query_params["tk"] = str(int(target_k))
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    st.session_state["n_points_slider"] = p["n_points"]
    st.session_state["k_slider"] = p["k"]
    st.session_state["spread_slider"] = p["spread"]
    st.session_state["size_imbalance_slider"] = p["size_imbalance"]
    st.session_state["bridge_strength_slider"] = p["bridge_strength"]
    st.session_state["shape_radio"] = p["shape"]
    st.session_state["linkage_radio"] = p["linkage"]
    st.session_state["target_k_slider"] = p["target_k"]
    st.session_state["seed_input"] = p["seed"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
