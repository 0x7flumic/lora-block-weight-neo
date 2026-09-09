"""
LoRA Block Weight Extension for ForgeNEO
Allows per-block weight control of LoRA models.

Supports:
  - Inline syntax:  <lora:name:1:lbw=1,0,0,...>   (named param, safe)
  - Inline syntax:  <lora:name:1:lbw=MIDD>         (preset name)
  - Positional:     <lora:name:1:1,0,0,...>         (monkeypatched to prevent crash)
  - UI panel with preset selector, weight editor, and profile management
"""

import logging
import os
import json
import re
import struct
from typing import Optional, Dict, List

import gradio as gr

import modules.scripts as scripts
from modules import shared

logger = logging.getLogger("lora_block_weight")


# ═══════════════════════════════════════════════════════════
#  ArchitectureProfile Engine
# ═══════════════════════════════════════════════════════════

class ArchitectureProfile:
    def __init__(
        self,
        name: str,
        display_name: str,
        block_count: int,
        block_names: Dict[str, int],
        presets: Dict[str, str],
        key_patterns: List[re.Pattern],
    ):
        self.name = name
        self.display_name = display_name
        self.block_count = block_count
        self.block_names = block_names
        self.presets = presets
        self.key_patterns = key_patterns

    def get_block_index(self, key: str) -> Optional[int]:
        for pat in self.key_patterns:
            m = pat.search(key)
            if m:
                if pat.groups >= 1:
                    return int(m.group(1))
                return None
        return None

    def resolve_block_name(self, name: str) -> Optional[List[float]]:
        idx = self.block_names.get(name.upper())
        if idx is not None and idx < self.block_count:
            vec = [0.0] * self.block_count
            vec[idx] = 1.0
            return vec
        return None


# Regex patterns for key detection (supports both standard dot and Kohya underscore separators)
_RE_SD_IN = re.compile(r"(?:input_blocks|down_blocks)[._](\d+)[._]")
_RE_SD_MID = re.compile(r"(?:middle_block|mid_block)[._]")
_RE_SD_OUT = re.compile(r"(?:output_blocks|up_blocks)[._](\d+)[._]")
_RE_SD_BASE = re.compile(r"(?:model[._])?(?:diffusion_model[._])?(time_embed|label_emb|out)[._]")

_RE_FLUX_DBL = re.compile(r"double_blocks[._](\d+)[._]")
_RE_FLUX_SNG = re.compile(r"single_blocks[._](\d+)[._]")

_RE_BLOCKS = re.compile(r"blocks[._](\d+)[._]")


# SD / SDXL (26 values: 0=BASE, 1-12=IN00-IN11, 13=M00, 14-25=OUT00-OUT11)
SDXL_PRESETS = {
    # Early down-blocks encode spatial layout and pose
    "COMPOSITION": "1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
    # Deep down-blocks, mid bottleneck, and early up-blocks capture identity and facial features
    "FACE":        "0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0",
    # Mid-to-late output blocks decode color palettes and aesthetic styling
    "STYLE":       "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1",
    # Deepest output blocks decode high-frequency lines and micro-textures
    "TEXTURE":     "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1",
}
SDXL_BLOCK_NAMES = {
    "BASE": 0,
    **{f"IN{i:02d}": i + 1 for i in range(12)},
    "M00": 13, "MID": 13, "MIDD": 13,
    **{f"OUT{i:02d}": i + 14 for i in range(12)},
}

def _sd_get_block_index(key: str) -> Optional[int]:
    m = _RE_SD_IN.search(key)
    if m:
        return int(m.group(1)) + 1
    if _RE_SD_MID.search(key):
        return 13
    m = _RE_SD_OUT.search(key)
    if m:
        return int(m.group(1)) + 14
    if _RE_SD_BASE.search(key):
        return 0
    return None

class SDXLProfile(ArchitectureProfile):
    def __init__(self):
        super().__init__(
            name="sdxl",
            display_name="SD/XL",
            block_count=26,
            block_names=SDXL_BLOCK_NAMES,
            presets=SDXL_PRESETS,
            key_patterns=[_RE_SD_IN, _RE_SD_MID, _RE_SD_OUT, _RE_SD_BASE]
        )
    def get_block_index(self, key: str) -> Optional[int]:
        return _sd_get_block_index(key)


# Flux.1 (57 values: 19 double + 38 single)
FLUX1_PRESETS = {
    "COMPOSITION": ",".join(["1"] * 10 + ["0"] * 9 + ["0"] * 38),
    "FACE":        ",".join(["0"] * 10 + ["1"] * 9 + ["1"] * 10 + ["0"] * 28),
    "STYLE":       ",".join(["0"] * 19 + ["0"] * 10 + ["1"] * 28),
    "TEXTURE":     ",".join(["0"] * 19 + ["0"] * 28 + ["1"] * 10),
}
FLUX1_BLOCK_NAMES = {
    **{f"D{i:02d}": i for i in range(19)},
    **{f"S{i:02d}": i + 19 for i in range(38)},
}

class Flux1Profile(ArchitectureProfile):
    def __init__(self, num_dbl=19, num_sng=38):
        super().__init__(
            name="flux1",
            display_name="Flux.1",
            block_count=num_dbl + num_sng,
            block_names=FLUX1_BLOCK_NAMES,
            presets=FLUX1_PRESETS,
            key_patterns=[_RE_FLUX_DBL, _RE_FLUX_SNG]
        )
        self.num_dbl = num_dbl
        self.num_sng = num_sng

    def get_block_index(self, key: str) -> Optional[int]:
        m = _RE_FLUX_DBL.search(key)
        if m:
            return int(m.group(1))
        m = _RE_FLUX_SNG.search(key)
        if m:
            return int(m.group(1)) + self.num_dbl
        return None


# Flux.2-Klein 9B (32 values: 8 double + 24 single)
FLUX_KLEIN_9B_PRESETS = {
    "COMPOSITION": ",".join(["1"] * 4 + ["0"] * 4 + ["0"] * 24),
    "FACE":        ",".join(["0"] * 4 + ["1"] * 4 + ["1"] * 6 + ["0"] * 18),
    "STYLE":       ",".join(["0"] * 8 + ["0"] * 6 + ["1"] * 18),
    "TEXTURE":     ",".join(["0"] * 8 + ["0"] * 18 + ["1"] * 6),
}
FLUX_KLEIN_9B_BLOCK_NAMES = {
    **{f"D{i:02d}": i for i in range(8)},
    **{f"S{i:02d}": i + 8 for i in range(24)},
}

class FluxKlein9BProfile(ArchitectureProfile):
    def __init__(self):
        super().__init__(
            name="flux_k9b",
            display_name="F2-K9B",
            block_count=32,
            block_names=FLUX_KLEIN_9B_BLOCK_NAMES,
            presets=FLUX_KLEIN_9B_PRESETS,
            key_patterns=[_RE_FLUX_DBL, _RE_FLUX_SNG]
        )
    def get_block_index(self, key: str) -> Optional[int]:
        m = _RE_FLUX_DBL.search(key)
        if m:
            return int(m.group(1))
        m = _RE_FLUX_SNG.search(key)
        if m:
            return int(m.group(1)) + 8
        return None


# Flux.2-Klein 4B (25 values: 5 double + 20 single)
FLUX_KLEIN_4B_PRESETS = {
    "COMPOSITION": ",".join(["1"] * 3 + ["0"] * 2 + ["0"] * 20),
    "FACE":        ",".join(["0"] * 3 + ["1"] * 2 + ["1"] * 5 + ["0"] * 15),
    "STYLE":       ",".join(["0"] * 5 + ["0"] * 5 + ["1"] * 15),
    "TEXTURE":     ",".join(["0"] * 5 + ["0"] * 15 + ["1"] * 5),
}
FLUX_KLEIN_4B_BLOCK_NAMES = {
    **{f"D{i:02d}": i for i in range(5)},
    **{f"S{i:02d}": i + 5 for i in range(20)},
}

class FluxKlein4BProfile(ArchitectureProfile):
    def __init__(self):
        super().__init__(
            name="flux_k4b",
            display_name="F2-K4B",
            block_count=25,
            block_names=FLUX_KLEIN_4B_BLOCK_NAMES,
            presets=FLUX_KLEIN_4B_PRESETS,
            key_patterns=[_RE_FLUX_DBL, _RE_FLUX_SNG]
        )
    def get_block_index(self, key: str) -> Optional[int]:
        m = _RE_FLUX_DBL.search(key)
        if m:
            return int(m.group(1))
        m = _RE_FLUX_SNG.search(key)
        if m:
            return int(m.group(1)) + 5
        return None


# Anima 2B (28 blocks: B00-B27)
ANIMA_2B_PRESETS = {
    "COMPOSITION": ",".join(["1"] * 7 + ["0"] * 21),
    "FACE":        ",".join(["0"] * 7 + ["1"] * 11 + ["0"] * 10),
    "STYLE":       ",".join(["0"] * 14 + ["1"] * 14),
    "TEXTURE":     ",".join(["0"] * 24 + ["1"] * 4),
}

# Anima 2.9B (40 blocks: B00-B39)
ANIMA_29B_PRESETS = {
    "COMPOSITION": ",".join(["1"] * 10 + ["0"] * 30),
    "FACE":        ",".join(["0"] * 10 + ["1"] * 16 + ["0"] * 14),
    "STYLE":       ",".join(["0"] * 20 + ["1"] * 20),
    "TEXTURE":     ",".join(["0"] * 34 + ["1"] * 6),
}

class AnimaProfile(ArchitectureProfile):
    def __init__(self, block_count=28):
        disp = "Anima 2.9B" if block_count >= 40 else "Anima 2B"
        presets = ANIMA_29B_PRESETS if block_count >= 40 else ANIMA_2B_PRESETS
        super().__init__(
            name=f"anima{block_count}",
            display_name=disp,
            block_count=block_count,
            block_names={f"B{i:02d}": i for i in range(block_count)},
            presets=presets,
            key_patterns=[_RE_BLOCKS]
        )
    def get_block_index(self, key: str) -> Optional[int]:
        m = _RE_BLOCKS.search(key)
        return int(m.group(1)) if m else None



# Wan 2.1 (14B: 40 blocks, 1.3B: 30 blocks)
WAN_14B_PRESETS = {
    "COMPOSITION": ",".join(["1"] * 10 + ["0"] * 30),
    "FACE":        ",".join(["0"] * 10 + ["1"] * 15 + ["0"] * 15),
    "STYLE":       ",".join(["0"] * 20 + ["1"] * 20),
    "TEXTURE":     ",".join(["0"] * 32 + ["1"] * 8),
}

WAN_13B_PRESETS = {
    "COMPOSITION": ",".join(["1"] * 8 + ["0"] * 22),
    "FACE":        ",".join(["0"] * 8 + ["1"] * 12 + ["0"] * 10),
    "STYLE":       ",".join(["0"] * 15 + ["1"] * 15),
    "TEXTURE":     ",".join(["0"] * 24 + ["1"] * 6),
}

class WanProfile(ArchitectureProfile):
    def __init__(self, block_count=40):
        name = f"wan21_{'14b' if block_count >= 40 else '13b'}"
        disp = "Wan2.1-14B" if block_count >= 40 else "Wan2.1-1.3B"
        presets = WAN_14B_PRESETS if block_count >= 40 else WAN_13B_PRESETS
        super().__init__(
            name=name,
            display_name=disp,
            block_count=block_count,
            block_names={f"B{i:02d}": i for i in range(block_count)},
            presets=dict(presets),
            key_patterns=[_RE_BLOCKS]
        )
    def get_block_index(self, key: str) -> Optional[int]:
        m = _RE_BLOCKS.search(key)
        if m:
            return int(m.group(1))
        if any(x in key for x in ("patch_embedding", "time_projection", "head.")):
            return 0
        return None


# Qwen-Image (60 blocks default, 32 blocks variant)
_RE_QWEN_BLOCKS = re.compile(r"transformer_blocks[._](\d+)[._]")

QWEN_PRESETS = {
    "COMPOSITION": ",".join(["1"] * 15 + ["0"] * 45),
    "FACE":        ",".join(["0"] * 15 + ["1"] * 25 + ["0"] * 20),
    "STYLE":       ",".join(["0"] * 30 + ["1"] * 30),
    "TEXTURE":     ",".join(["0"] * 50 + ["1"] * 10),
}

class QwenImageProfile(ArchitectureProfile):
    def __init__(self, block_count=60):
        super().__init__(
            name="qwen_image",
            display_name="Qwen-Image",
            block_count=block_count,
            block_names={f"B{i:02d}": i for i in range(block_count)},
            presets=QWEN_PRESETS,
            key_patterns=[_RE_QWEN_BLOCKS]
        )
    def get_block_index(self, key: str) -> Optional[int]:
        m = _RE_QWEN_BLOCKS.search(key)
        if m:
            return int(m.group(1))
        if any(x in key for x in ("img_in", "txt_in", "norm_out", "proj_out", "time_text_embed")):
            return 0
        return None


PROFILE_REGISTRY: Dict[str, ArchitectureProfile] = {
    "sd15": SDXLProfile(),
    "sdxl": SDXLProfile(),
    "flux1": Flux1Profile(),
    "flux_k9b": FluxKlein9BProfile(),
    "flux_k4b": FluxKlein4BProfile(),
    "anima28": AnimaProfile(28),
    "anima40": AnimaProfile(40),
    "wan21_14b": WanProfile(40),
    "wan21_13b": WanProfile(30),
    "qwen_image": QwenImageProfile(60),
}


# ═══════════════════════════════════════════════════════════
#  User Presets Persistence
# ═══════════════════════════════════════════════════════════

def _get_user_presets_file() -> str:
    base = getattr(shared, "data_path", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base, "extensions", "lora-block-weight-neo")
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, "lbw_presets.json")

USER_PRESETS_FILE = _get_user_presets_file()

def _load_user_presets():
    if os.path.exists(USER_PRESETS_FILE):
        try:
            with open(USER_PRESETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    if k in PROFILE_REGISTRY and isinstance(v, dict):
                        PROFILE_REGISTRY[k].presets.update(v)
                if "sdxl" in data and "sd15" in PROFILE_REGISTRY:
                    PROFILE_REGISTRY["sd15"].presets.update(data["sdxl"])
                if "flux" in data and "flux1" in PROFILE_REGISTRY:
                    PROFILE_REGISTRY["flux1"].presets.update(data["flux"])
                if "anima" in data:
                    if "anima28" in PROFILE_REGISTRY:
                        PROFILE_REGISTRY["anima28"].presets.update(data["anima"])
                    if "anima40" in PROFILE_REGISTRY:
                        PROFILE_REGISTRY["anima40"].presets.update(data["anima"])
                if "wan" in data:
                    if "wan21_14b" in PROFILE_REGISTRY:
                        PROFILE_REGISTRY["wan21_14b"].presets.update(data["wan"])
                    if "wan21_13b" in PROFILE_REGISTRY:
                        PROFILE_REGISTRY["wan21_13b"].presets.update(data["wan"])
        except Exception as e:
            logger.error(f"[LBW] Failed to load user presets: {e}")

_load_user_presets()


# ═══════════════════════════════════════════════════════════
#  Architecture Detection & Key Mapping
# ═══════════════════════════════════════════════════════════

def detect_model_profile(patch_keys) -> ArchitectureProfile:
    """Detect architecture profile with exact block count inspection."""
    keys_set = set(patch_keys)

    # 1. SD / SDXL detection
    for k in keys_set:
        if _RE_SD_IN.search(k) or _RE_SD_OUT.search(k) or _RE_SD_MID.search(k):
            return PROFILE_REGISTRY["sdxl"]

    # 2. Flux variants detection
    dbl_max = -1
    sng_max = -1
    has_flux = False
    for k in keys_set:
        m_dbl = _RE_FLUX_DBL.search(k)
        if m_dbl:
            dbl_max = max(dbl_max, int(m_dbl.group(1)))
            has_flux = True
        m_sng = _RE_FLUX_SNG.search(k)
        if m_sng:
            sng_max = max(sng_max, int(m_sng.group(1)))
            has_flux = True

    if has_flux:
        num_dbl = dbl_max + 1 if dbl_max != -1 else 0
        num_sng = sng_max + 1 if sng_max != -1 else 0
        if num_dbl == 8 and num_sng == 24:
            return PROFILE_REGISTRY["flux_k9b"]
        elif num_dbl == 5 and num_sng == 20:
            return PROFILE_REGISTRY["flux_k4b"]
        elif num_dbl == 19 and num_sng == 38:
            return PROFILE_REGISTRY["flux1"]
        else:
            return Flux1Profile(num_dbl=max(1, num_dbl), num_sng=max(0, num_sng))

    # 3. Qwen-Image detection
    qwen_max = -1
    for k in keys_set:
        m = _RE_QWEN_BLOCKS.search(k)
        if m:
            qwen_max = max(qwen_max, int(m.group(1)))
    if qwen_max != -1:
        num_blocks = qwen_max + 1
        return QwenImageProfile(num_blocks)

    # 4. Wan 2.1 vs Anima detection
    is_wan = any("patch_embedding" in k or "time_projection" in k or "head." in k for k in keys_set)
    block_max = -1
    for k in keys_set:
        m = _RE_BLOCKS.search(k)
        if m:
            block_max = max(block_max, int(m.group(1)))

    if is_wan:
        num_blocks = block_max + 1 if block_max != -1 else 40
        return PROFILE_REGISTRY["wan21_14b"] if num_blocks >= 40 else PROFILE_REGISTRY["wan21_13b"]

    if block_max != -1:
        num_blocks = block_max + 1
        if num_blocks > 28:
            return PROFILE_REGISTRY["anima40"] if num_blocks <= 40 else AnimaProfile(num_blocks)
        return PROFILE_REGISTRY["anima28"]

    return PROFILE_REGISTRY["sdxl"]


def inspect_safetensors_data(filepath: str):
    """Read header tensor keys and metadata dict of a .safetensors file in <1ms."""
    try:
        if not os.path.isfile(filepath):
            return [], {}
        with open(filepath, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                return [], {}
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            if header_size <= 0 or header_size > 67108864:  # Safety cap at 64MB
                return [], {}
            header_json_bytes = f.read(header_size)
            header = json.loads(header_json_bytes.decode("utf-8"))
            meta = header.get("__metadata__", {})
            keys = [k for k in header.keys() if k != "__metadata__"]
            return keys, meta
    except Exception as e:
        logger.debug(f"[LBW] Could not inspect safetensors header for {filepath}: {e}")
        return [], {}


_lora_filepath_cache: Dict[str, str] = {}


def resolve_lora_filepath(name: str) -> Optional[str]:
    """Resolve LoRA name or alias to absolute filepath via Forge networks registry and disk fallback."""
    if not name:
        return None

    if name in _lora_filepath_cache:
        cached = _lora_filepath_cache[name]
        if os.path.isfile(cached):
            return cached

    base = os.path.splitext(os.path.basename(name))[0]
    if base in _lora_filepath_cache:
        cached = _lora_filepath_cache[base]
        if os.path.isfile(cached):
            return cached

    # 1. Try Forge networks module registry
    try:
        try:
            import networks as networks_module
        except ImportError:
            from extensions_builtin.sd_forge_lora import networks as networks_module

        net = networks_module.available_networks.get(name) or networks_module.available_network_aliases.get(name)
        if net is None:
            net = networks_module.available_networks.get(base) or networks_module.available_network_aliases.get(base)
        if net is not None and getattr(net, "filename", None) and os.path.isfile(net.filename):
            _lora_filepath_cache[name] = net.filename
            _lora_filepath_cache[base] = net.filename
            return net.filename

        lower_name = name.lower()
        lower_base = base.lower()
        for k, v in networks_module.available_networks.items():
            k_low = k.lower()
            if (k_low == lower_name or k_low == lower_base) and getattr(v, "filename", None) and os.path.isfile(v.filename):
                _lora_filepath_cache[name] = v.filename
                _lora_filepath_cache[base] = v.filename
                return v.filename
        for k, v in networks_module.available_network_aliases.items():
            k_low = k.lower()
            if (k_low == lower_name or k_low == lower_base) and getattr(v, "filename", None) and os.path.isfile(v.filename):
                _lora_filepath_cache[name] = v.filename
                _lora_filepath_cache[base] = v.filename
                return v.filename
    except Exception as e:
        logger.debug(f"[LBW] networks_module lookup error: {e}")

    # 2. Search configured LoRA directories on disk as fallback
    try:
        lower_base = base.lower()
        lora_dirs = []
        cmd_opts = getattr(shared, "cmd_opts", None)
        if cmd_opts:
            if getattr(cmd_opts, "lora_dir", None):
                lora_dirs.append(cmd_opts.lora_dir)
            if getattr(cmd_opts, "lora_dirs", None):
                lora_dirs.extend(cmd_opts.lora_dirs)
        default_dir = os.path.join(getattr(shared, "models_path", "models"), "Lora")
        if default_dir not in lora_dirs:
            lora_dirs.append(default_dir)

        for l_dir in lora_dirs:
            if not os.path.isdir(l_dir):
                continue
            for root, _, files in os.walk(l_dir):
                for f in files:
                    if f.endswith(".safetensors") and os.path.splitext(f)[0].lower() == lower_base:
                        full_path = os.path.join(root, f)
                        _lora_filepath_cache[name] = full_path
                        _lora_filepath_cache[base] = full_path
                        return full_path
    except Exception as e:
        logger.debug(f"[LBW] Error resolving LoRA filepath on disk for '{name}': {e}")

    return None


_detected_arch_cache: Dict[str, str] = {}


def detect_lora_arch_by_name(name: str) -> Optional[ArchitectureProfile]:
    """Inspect safetensors keys and metadata from disk to determine exact architecture profile."""
    if not name:
        return None
    if name in _detected_arch_cache:
        prof_key = _detected_arch_cache[name]
        return PROFILE_REGISTRY.get(prof_key)

    filepath = resolve_lora_filepath(name)
    if not filepath or not filepath.endswith(".safetensors"):
        return None

    keys, meta = inspect_safetensors_data(filepath)
    if not keys and not meta:
        return None

    # Check explicit metadata hints (e.g. ss_base_model_version: anima)
    meta_arch = str(meta.get("ss_base_model_version", "")).lower()
    modelspec_arch = str(meta.get("modelspec.architecture", "")).lower()
    is_meta_anima = "anima" in meta_arch or "anima" in modelspec_arch

    profile = detect_model_profile(keys)

    # If keys detection was ambiguous or defaulted to SDXL but metadata explicitly declares anima
    if (profile is None or profile.name == "sdxl") and is_meta_anima:
        block_max = -1
        for k in keys:
            m = _RE_BLOCKS.search(k)
            if m:
                block_max = max(block_max, int(m.group(1)))
        total_blocks = block_max + 1 if block_max != -1 else 28
        profile = PROFILE_REGISTRY["anima40"] if total_blocks > 28 else PROFILE_REGISTRY["anima28"]

    if profile:
        _detected_arch_cache[name] = profile.name
    return profile


def adapt_block_weights(weights: List[float], target_len: int, arch: str = "sd") -> List[float]:
    """Adapt or interpolate weight vectors to match target block length."""
    src_len = len(weights)
    if src_len == target_len:
        return weights

    # SD 17-block to 26-block expansion
    if src_len == 17 and target_len == 26 and "sd" in arch:
        indices_17 = [0, 2, 3, 5, 6, 8, 9, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        expanded = [weights[0]] * 26
        for val, idx in zip(weights, indices_17):
            expanded[idx] = val
        return expanded

    # SDXL 12-block to 26-block expansion
    if src_len == 12 and target_len == 26 and "sd" in arch:
        indices_12 = [0, 5, 6, 8, 9, 13, 14, 15, 16, 17, 18, 19]
        expanded = [weights[0]] * 26
        for val, idx in zip(weights, indices_12):
            expanded[idx] = val
        return expanded

    # Flux 57-block adapt to Klein
    if src_len == 57 and "flux" in arch:
        dbl_src = weights[:19]
        sng_src = weights[19:57]
        if target_len == 32:  # 8 dbl + 24 sng
            return dbl_src[:8] + sng_src[:24]
        if target_len == 25:  # 5 dbl + 20 sng
            return dbl_src[:5] + sng_src[:20]

    # Linear interpolation for continuous architectures (Anima, Wan)
    if src_len > 1 and target_len > 1:
        resampled = []
        for i in range(target_len):
            src_pos = i * (src_len - 1) / (target_len - 1)
            lower = int(src_pos)
            upper = min(lower + 1, src_len - 1)
            frac = src_pos - lower
            val = (1.0 - frac) * weights[lower] + frac * weights[upper]
            resampled.append(val)
        return resampled

    # Fallback pad or slice
    if src_len < target_len:
        return weights + [1.0] * (target_len - src_len)
    return weights[:target_len]


# Legacy preset alias mappings for backward compatibility
LEGACY_PRESET_ALIASES: Dict[str, str] = {
    "MIDD": "FACE",
    "MID": "FACE",
    "M": "FACE",
    "CHARACTER": "FACE",
    "INS": "COMPOSITION",
    "IN": "COMPOSITION",
    "LAYOUT": "COMPOSITION",
    "OUTS": "STYLE",
    "OUTD": "TEXTURE",
}

def parse_block_weights(raw_str: str, profile: Optional[ArchitectureProfile] = None) -> Optional[List[float]]:
    """Parse block weights via active profile."""
    raw_str = str(raw_str).strip()
    if not raw_str:
        return None

    if profile is None:
        profile = PROFILE_REGISTRY["sdxl"]

    upper = raw_str.upper()

    # Resolve backward-compatibility aliases first
    if upper in LEGACY_PRESET_ALIASES:
        upper = LEGACY_PRESET_ALIASES[upper]

    # 1. Preset lookup
    if upper in profile.presets:
        return [float(x.strip()) for x in profile.presets[upper].split(",")]

    # 2. Check other profiles presets (cross-preset fallback)
    for p in PROFILE_REGISTRY.values():
        if upper in p.presets:
            vec = [float(x.strip()) for x in p.presets[upper].split(",")]
            return adapt_block_weights(vec, profile.block_count, arch=profile.name)

    # 3. Single block alias
    result = profile.resolve_block_name(upper)
    if result is not None:
        return result

    # 4. Comma-separated floats
    parts = raw_str.split(",")
    try:
        parsed = [float(x.strip()) for x in parts]
        return adapt_block_weights(parsed, profile.block_count, arch=profile.name)
    except ValueError:
        logger.warning(f"[LBW] Could not parse block weights: {raw_str}")
        return None


def _is_block_weight_string(val) -> bool:
    if not isinstance(val, str):
        return False
    val_str = val.strip()
    if not val_str:
        return False
    # Comma-separated numbers
    if "," in val_str and re.match(r"^[\d.,\s\-+eE]+$", val_str):
        return True
    val_upper = val_str.upper()
    if val_upper in LEGACY_PRESET_ALIASES:
        return True
    for prof in PROFILE_REGISTRY.values():
        if val_upper in prof.presets or val_upper in prof.block_names:
            return True
    return False


# ═══════════════════════════════════════════════════════════
#  Global State
# ═══════════════════════════════════════════════════════════
_extension_enabled = False
_active_block_weights = {}
_last_lbw_state = None

_original_activate = None
_original_load_lora = None
_patches_installed = False


def _install_patches():
    """Install monkeypatches on Forge's ExtraNetworkLora and networks loader."""
    global _patches_installed, _original_activate, _original_load_lora

    if _patches_installed:
        return

    logger.debug("[LBW] Starting _install_patches...")

    try:
        from modules.extra_networks import extra_network_registry
        lora_net = extra_network_registry.get("lora")
        if lora_net is not None and _original_activate is None:
            _original_activate = lora_net.activate
            lora_net.activate = _patched_activate.__get__(lora_net, type(lora_net))
            logger.info("[LBW] Patched ExtraNetworkLora.activate")
    except Exception as e:
        logger.error(f"[LBW] Failed to patch ExtraNetworkLora.activate: {e}")
        return

    try:
        import networks as networks_module
        if _original_load_lora is None:
            _original_load_lora = networks_module.load_lora_for_models
            networks_module.load_lora_for_models = _patched_load_lora_for_models
            logger.info("[LBW] Patched networks.load_lora_for_models")
    except Exception as e:
        logger.error(f"[LBW] Failed to patch networks.load_lora_for_models: {e}")
        return

    _patches_installed = True
    logger.info("[LBW] All patches installed successfully")


# ═══════════════════════════════════════════════════════════
#  Patched ExtraNetworkLora.activate
# ═══════════════════════════════════════════════════════════

def _patched_activate(self, p, params_list):
    global _active_block_weights, _last_lbw_state

    def _call_original(args_list):
        if hasattr(_original_activate, "__self__"):
            return _original_activate(p, args_list)
        return _original_activate(self, p, args_list)

    if not params_list:
        return _call_original(params_list)

    for params in params_list:
        if not getattr(params, "positional", None):
            continue

        name = params.positional[0]
        block_weight_str = None

        # 1. Named parameter lbw=...
        if "lbw" in params.named:
            block_weight_str = params.named.pop("lbw")

        # 2. Positional cleaning
        clean_positional = []
        for i, val in enumerate(params.positional):
            if i >= 2 and _is_block_weight_string(val):
                if not block_weight_str or not _is_block_weight_string(block_weight_str):
                    block_weight_str = str(val)
            else:
                clean_positional.append(val)

        params.positional = clean_positional
        params.items = params.positional + [f"{k}={v}" for k, v in params.named.items()]

        if block_weight_str and _extension_enabled:
            _active_block_weights[name] = block_weight_str
            logger.info(f"[LBW] {name}: queued block weight string '{block_weight_str}'")

    current_state = str(_active_block_weights)
    if current_state != _last_lbw_state:
        _last_lbw_state = current_state
        _clear_forge_lora_cache()

    return _call_original(params_list)


def _clear_forge_lora_cache():
    cleared = False
    try:
        sd_model = getattr(shared, "sd_model", None)
        if sd_model is not None and hasattr(sd_model, "current_lora_hash"):
            sd_model.current_lora_hash = None
            cleared = True
        from modules import sd_models
        model_data = getattr(sd_models, "model_data", None)
        if model_data is not None:
            current_sd = model_data.get_sd_model()
            if current_sd is not None:
                current_sd.current_lora_hash = None
                cleared = True
    except Exception as e:
        logger.error(f"[LBW] Error while clearing Forge LoRA cache: {e}")
    if cleared:
        logger.info("[LBW] Block weights changed; cleared Forge LoRA cache to force reload.")


# ═══════════════════════════════════════════════════════════
#  Patched load_lora_for_models
# ═══════════════════════════════════════════════════════════

def resolve_lora_weights(lora_basename: str, filename: str, active_weights: Dict[str, str]) -> Optional[str]:
    """Resolve active block weights for a LoRA with exact matching (no substring leakage)."""
    if not active_weights:
        return None

    # Step 1: Exact mapping via Forge's internal network dictionaries
    try:
        import networks as networks_module
        current_path = os.path.normpath(os.path.abspath(filename)).lower()
        for name, weights in active_weights.items():
            net_on_disk = networks_module.available_network_aliases.get(name) or networks_module.available_networks.get(name)
            if net_on_disk is not None and getattr(net_on_disk, "filename", None):
                if os.path.normpath(os.path.abspath(net_on_disk.filename)).lower() == current_path:
                    return weights
    except Exception:
        pass

    # Step 2: Exact name match
    if lora_basename in active_weights:
        return active_weights[lora_basename]

    # Step 3: Case-insensitive exact name match
    lower = lora_basename.lower()
    for name, weights in active_weights.items():
        if name.lower() == lower:
            return weights

    return None


def _get_patcher(obj):
    """Retrieve the underlying ModelPatcher instance if wrapped (e.g. CLIP.patcher)."""
    if obj is None:
        return None
    if hasattr(obj, "patcher") and hasattr(obj.patcher, "patches"):
        return obj.patcher
    return obj


def _patched_load_lora_for_models(model, clip, lora, strength_model, strength_clip,
                                   filename="default", online_mode=False):
    """Load LoRA and scale its model patches without tuple unpacking crashes."""
    if not _active_block_weights:
        return _original_load_lora(model, clip, lora, strength_model, strength_clip,
                                   filename, online_mode)

    # Snapshot patch counts before loading
    old_lengths = {}
    old_clip_lengths = {}
    try:
        model_patcher = _get_patcher(model)
        if model_patcher is not None:
            source_patches = model_patcher.weight_wrapper_patches if online_mode else model_patcher.patches
            for k, v in source_patches.items():
                old_lengths[k] = len(v)

        clip_patcher = _get_patcher(clip)
        if clip_patcher is not None:
            source_clip_patches = clip_patcher.weight_wrapper_patches if online_mode else clip_patcher.patches
            for k, v in source_clip_patches.items():
                old_clip_lengths[k] = len(v)
    except Exception as e:
        logger.warning(f"[LBW] Failed to snapshot patch lengths: {e}")

    new_model, new_clip = _original_load_lora(
        model, clip, lora, strength_model, strength_clip, filename, online_mode
    )

    lora_basename = os.path.splitext(os.path.basename(filename))[0]
    block_weights_str = resolve_lora_weights(lora_basename, filename, _active_block_weights)

    if block_weights_str is None:
        return new_model, new_clip

    try:
        new_model_patcher = _get_patcher(new_model)
        if new_model_patcher is None:
            return new_model, new_clip

        # Detect architecture profile from new patches
        patch_keys = set(new_model_patcher.weight_wrapper_patches.keys() if online_mode else new_model_patcher.patches.keys())
        profile = detect_model_profile(patch_keys)
        block_weights = parse_block_weights(block_weights_str, profile=profile)

        if block_weights is None:
            return new_model, new_clip

        scaled_count = 0

        if online_mode:
            # Scale OnlineLoRAPatch wrappers
            for key, wrappers in new_model_patcher.weight_wrapper_patches.items():
                old_len = old_lengths.get(key, 0)
                if len(wrappers) <= old_len:
                    continue
                b_idx = profile.get_block_index(key)
                scale = 1.0 if (b_idx is None or b_idx >= len(block_weights)) else float(block_weights[b_idx])
                for i in range(old_len, len(wrappers)):
                    w = wrappers[i]
                    if hasattr(w, "patch") and isinstance(w.patch, list) and len(w.patch) > 0:
                        pt = w.patch[0]
                        if isinstance(pt, list) and len(pt) >= 1:
                            pt[0] = pt[0] * scale
                            scaled_count += 1
        else:
            # Scale offline patches: dynamic tuple slicing
            for key, patches in new_model_patcher.patches.items():
                old_len = old_lengths.get(key, 0)
                if len(patches) <= old_len:
                    continue
                b_idx = profile.get_block_index(key)
                scale = 1.0 if (b_idx is None or b_idx >= len(block_weights)) else float(block_weights[b_idx])
                for i in range(old_len, len(patches)):
                    pt = patches[i]
                    patches[i] = (pt[0] * scale, *pt[1:])
                    scaled_count += 1

        # 2. Scale Text Encoder (CLIP) patches using BASE weight
        new_clip_patcher = _get_patcher(new_clip)
        if new_clip_patcher is not None and len(block_weights) > 0:
            base_scale = float(block_weights[0])
            if online_mode:
                for key, wrappers in new_clip_patcher.weight_wrapper_patches.items():
                    old_len = old_clip_lengths.get(key, 0)
                    for i in range(old_len, len(wrappers)):
                        w = wrappers[i]
                        if hasattr(w, "patch") and isinstance(w.patch, list) and len(w.patch) > 0:
                            pt = w.patch[0]
                            if isinstance(pt, list) and len(pt) >= 1:
                                pt[0] = pt[0] * base_scale
            else:
                for key, patches in new_clip_patcher.patches.items():
                    old_len = old_clip_lengths.get(key, 0)
                    for i in range(old_len, len(patches)):
                        pt = patches[i]
                        patches[i] = (pt[0] * base_scale, *pt[1:])

        logger.info(f"[LBW] {lora_basename}: scaled {scaled_count} patches ({profile.display_name})")
    except Exception as e:
        logger.error(f"[LBW] Error while applying block weights to {lora_basename}: {e}", exc_info=True)

    return new_model, new_clip


# ═══════════════════════════════════════════════════════════
#  UI & Gradio Controls
# ═══════════════════════════════════════════════════════════

def _save_preset_internal(prof_key: str, name: str, weights_str: str) -> bool:
    """Save preset to USER_PRESETS_FILE and reload engine state."""
    try:
        data = {}
        if os.path.exists(USER_PRESETS_FILE):
            with open(USER_PRESETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        data.setdefault(prof_key, {})[name] = weights_str.strip()
        with open(USER_PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        _load_user_presets()
        return True
    except Exception as e:
        logger.error(f"[LBW] Failed to save preset: {e}")
        return False


def _delete_preset_internal(prof_key: str, name: str) -> bool:
    """Delete preset from USER_PRESETS_FILE and reload engine state."""
    try:
        if not os.path.exists(USER_PRESETS_FILE):
            return True
        with open(USER_PRESETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Remove from prof_key if present
        removed = False
        if prof_key in data and name in data[prof_key]:
            del data[prof_key][name]
            removed = True
        
        # Also clean aliases if applicable
        if prof_key in ("anima28", "anima40") and "anima" in data and name in data["anima"]:
            del data["anima"][name]
            removed = True
        if prof_key in ("wan21_14b", "wan21_13b") and "wan" in data and name in data["wan"]:
            del data["wan"][name]
            removed = True
            
        if removed:
            with open(USER_PRESETS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        
        # Reset profile preset dictionaries and reload
        keys_to_clean = [prof_key]
        if prof_key in ("anima28", "anima40"):
            keys_to_clean.extend(["anima28", "anima40"])
        elif prof_key in ("wan21_14b", "wan21_13b"):
            keys_to_clean.extend(["wan21_14b", "wan21_13b"])
        elif prof_key in ("sdxl", "sd15"):
            keys_to_clean.extend(["sdxl", "sd15"])

        for k in set(keys_to_clean):
            if k in PROFILE_REGISTRY and name in PROFILE_REGISTRY[k].presets:
                del PROFILE_REGISTRY[k].presets[name]

        _load_user_presets()
        return True
    except Exception as e:
        logger.error(f"[LBW] Failed to delete preset: {e}")
        return False


def _get_all_presets_dict() -> Dict[str, Dict[str, str]]:
    """Return all presets grouped by architecture."""
    out = {}
    for prof_name, prof in PROFILE_REGISTRY.items():
        out[prof_name] = dict(prof.presets)
    return out


# Register API endpoints with FastAPI on WebUI launch
try:
    from modules import script_callbacks
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    def _on_app_started(demo: Optional[gr.Blocks], app: FastAPI):
        @app.get("/lbw/api/presets")
        async def api_get_presets():
            return JSONResponse({"presets": _get_all_presets_dict()})

        @app.get("/lbw/api/detect_lora")
        async def api_detect_lora(name: str = ""):
            name = name.strip()
            if not name:
                return JSONResponse({"arch": None, "error": "LoRA name required"}, status_code=400)
            try:
                profile = detect_lora_arch_by_name(name)
                if profile is not None:
                    return JSONResponse({
                        "arch": profile.name,
                        "block_count": profile.block_count,
                        "display_name": profile.display_name
                    })
                return JSONResponse({"arch": None})
            except Exception as e:
                logger.debug(f"[LBW] Error detecting architecture for '{name}': {e}")
                return JSONResponse({"arch": None, "error": str(e)}, status_code=500)

        @app.post("/lbw/api/presets")
        async def api_save_preset(req: Request):
            try:
                body = await req.json()
                arch = body.get("arch", "sdxl")
                name = body.get("name", "").strip()
                weights = body.get("weights", "").strip()
                if not name or not weights:
                    return JSONResponse({"success": False, "error": "Name and weights are required"}, status_code=400)
                
                # Map arch to profile key
                if arch in ("anima", "anima28"):
                    prof_key = "anima28"
                elif arch == "anima40":
                    prof_key = "anima40"
                elif arch in ("flux", "flux1"):
                    prof_key = "flux1"
                elif arch in ("flux_k9b", "flux_k4b", "wan21_14b", "wan21_13b", "qwen_image", "sdxl"):
                    prof_key = arch
                else:
                    prof_key = "sdxl"

                success = _save_preset_internal(prof_key, name, weights)
                return JSONResponse({"success": success, "presets": _get_all_presets_dict()})
            except Exception as e:
                return JSONResponse({"success": False, "error": str(e)}, status_code=500)

        @app.delete("/lbw/api/presets")
        async def api_delete_preset(req: Request):
            try:
                body = await req.json()
                arch = body.get("arch", "sdxl")
                name = body.get("name", "").strip()
                if not name:
                    return JSONResponse({"success": False, "error": "Preset name is required"}, status_code=400)

                # Map arch to profile key
                if arch in ("anima", "anima28"):
                    prof_key = "anima28"
                elif arch == "anima40":
                    prof_key = "anima40"
                elif arch in ("flux", "flux1"):
                    prof_key = "flux1"
                elif arch in ("flux_k9b", "flux_k4b", "wan21_14b", "wan21_13b", "qwen_image", "sdxl"):
                    prof_key = arch
                else:
                    prof_key = "sdxl"

                success = _delete_preset_internal(prof_key, name)
                return JSONResponse({"success": success, "presets": _get_all_presets_dict()})
            except Exception as e:
                return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    script_callbacks.on_app_started(_on_app_started)
except Exception as e:
    logger.debug(f"[LBW] script_callbacks setup skipped: {e}")


class LoraBlockWeight(scripts.Script):
    def title(self):
        return "LoRA Block Weight Neo"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        try:
            from modules.ui_components import InputAccordion
            acc_context = InputAccordion(False, label="LoRA Block Weight Neo", elem_id=self.elem_id("enable"))
        except Exception:
            acc_context = None

        markdown_info = (
            "**How to use:** Left-click or place cursor inside any `<lora:name:...>` tag in your prompt to open the floating menu to configure presets, adjust block sliders, and manage custom weights.\n\n"
            "**Prompt Syntax:** `<lora:name:weight:lbw=preset_or_vector>`\n\n"
            "* **Built-in Presets:** `COMPOSITION`, `FACE`, `STYLE`, `TEXTURE`, `RESET`\n"
            "* **Direct Block Vector:** Comma-separated floats (e.g. `1,1,0,0,...`)\n\n"
            "**Supported Architectures:**\n"
            "* **SD 1.5 & SDXL (26):** `BASE`, `IN00-IN11`, `M00`, `OUT00-OUT11`\n"
            "* **Flux.1 (57):** `D00-D18` (double), `S00-S37` (single)\n"
            "* **Flux.2-Klein 9B (32) / 4B (25):** `D00-D07` / `D00-D04` (double), `S00-S23` / `S00-S19` (single)\n"
            "* **Anima 2B (28) / 2.9B (40):** `B00-B27` / `B00-B39`\n"
            "* **Wan 2.1-14B (40) / 1.3B (30):** `B00-B39` / `B00-B29`\n"
            "* **Qwen-Image (60):** `B00-B59`"
        )

        if acc_context is not None:
            with acc_context as enabled:
                gr.Markdown(markdown_info)
        else:
            with gr.Accordion("LoRA Block Weight Neo", open=False):
                enabled = gr.Checkbox(label="Enable LoRA Block Weight", value=False)
                gr.Markdown(markdown_info)

        return [enabled]

    def process(self, p, enabled, *args, **kwargs):
        global _extension_enabled, _active_block_weights
        _extension_enabled = bool(enabled)
        _active_block_weights.clear()
        _install_patches()

    def postprocess(self, p, processed, *args):
        global _active_block_weights
        _active_block_weights.clear()
