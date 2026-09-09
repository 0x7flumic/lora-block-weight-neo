# LoRA Block Weight Neo

A **Stable Diffusion WebUI Forge (Neo)** extension enabling dynamic, block-level weight control for LoRA models across **SD 1.5, SDXL, Pony, Illustrious, NoobAI, Flux.1, Flux.2-Klein, Anima, Wan 2.1, and Qwen-Image** architectures.

Fine-tune how much character likeness, composition, art style, aesthetic coloring, or fine texture from a LoRA affects your generations on a block-by-block basis.

---

## Key Features

- **Decoupled Architecture Engine:**
  - **SD 1.5 & SDXL / Pony / Illustrious / NoobAI (26 Blocks):** Full U-Net skeleton (`BASE`, `IN00-IN11`, `M00`, `OUT00-OUT11`) with automated legacy 17-block and 12-block vector expansion.
  - **Flux.1 Dev / Schnell (57 Blocks):** Full double-stream (`D00-D18`) and single-stream (`S00-S37`) block control.
  - **Flux.2-Klein 9B (32 Blocks):** `D00-D07` double stream + `S00-S23` single stream.
  - **Flux.2-Klein 4B (25 Blocks):** `D00-D04` double stream + `S00-S19` single stream.
  - **Anima 2B (28 Blocks) & Anima 2.9B (40 Blocks):** Dedicated DiT block stacks (`B00-B27` and `B00-B39`).
  - **Wan 2.1 Video & Image DiT:** Distinct profiles for **Wan 14B** (40 blocks: `B00-B39`) and **Wan 1.3B** (30 blocks: `B00-B29`).
  - **Qwen-Image (60 Blocks):** Full DiT transformer stack (`B00-B59`).
- **Interactive In-Prompt Floating Popover:**
  - **Instant Trigger:** Simply left-click or place the text cursor inside any `<lora:name:...>` tag in your prompt.
  - **Metadata Auto-Detection:** Automatically queries Extra Networks card metadata and safetensors headers to identify the exact architecture (Not always accurate).
  - **Live LoRA Multiplier Slider:** Adjust the overall LoRA strength multiplier in real time without touching your block weights.
  - **One-Click Presets:** Apply presets (`COMPOSITION`, `FACE`, `STYLE`, `TEXTURE`, `RESET`) instantly.
  - **Custom Preset Manager:** Save named block presets directly to disk (`lbw_presets.json`) and manage or delete them from the inline `Manage ⚙` panel.
- **Fail-Proof Engine:**
  - Graceful fallback: If an unexpected layer configuration or mismatch occurs, the LoRA safely loads with standard weights instead of failing generation.
  - Cross-architecture linear interpolation: Seamlessly applies presets across different model architectures and block counts.
  - Text encoder (CLIP) scaling: Preserves prompt alignment by scaling CLIP patches with the root `BASE` weight.
  - Automatic cache clearing: Automatically triggers Forge LoRA cache invalidation when weights change so adjustments take effect immediately.

---

## Supported Architectures & Block Specifications

| Architecture | Block Count | Block Index / Layer Mappings |
| :--- | :---: | :--- |
| **SD 1.5 & SDXL (Full U-Net)** | **26** | `BASE` (0), `IN00-IN11` (1-12), `M00` (13), `OUT00-OUT11` (14-25) |
| **SD 1.5 (Legacy Shorthand)** | **17** | Traditional SD 1.5 attention layers (auto-expanded to 26 blocks) |
| **SDXL / Pony / Illustrious (Legacy)** | **12** | Traditional SDXL attention layers (auto-expanded to 26 blocks) |
| **Flux.1 (Dev / Schnell)** | **57** | Double blocks `D00-D18` (0-18) + Single blocks `S00-S37` (19-56) |
| **Flux.2-Klein 9B** | **32** | Double blocks `D00-D07` (0-7) + Single blocks `S00-S23` (8-31) |
| **Flux.2-Klein 4B** | **25** | Double blocks `D00-D04` (0-4) + Single blocks `S00-S19` (5-24) |
| **Anima 2B** | **28** | DiT block stack `B00-B27` |
| **Anima 2.9B** | **40** | DiT block stack `B00-B39` |
| **Wan 2.1 14B** | **40** | Video DiT stack `B00-B39` |
| **Wan 2.1 1.3B** | **30** | Video DiT stack `B00-B29` |
| **Qwen-Image** | **60** | Transformer block stack `B00-B59` |

---

## Prompt Syntax Guide

### 1. Named Preset Syntax (Recommended)
Use the named parameter `lbw=preset_name`:
```text
<lora:my_lora_name:1:lbw=FACE>
<lora:my_lora_name:0.85:lbw=STYLE>
<lora:my_lora_name:1:lbw=COMPOSITION>
<lora:my_lora_name:1:lbw=TEXTURE>
```
*(Legacy aliases `MIDD`, `MID`, `INS`, and `OUTS` are automatically mapped for backward compatibility.)*

### 2. Custom User Preset Syntax
Saved custom presets can be referenced directly by name:
```text
<lora:my_lora_name:1:lbw=MyCustomPreset>
```

### 3. Comma-Separated Block Vector Syntax
Specify exact floats for every block in the architecture:
```text
<lora:my_lora_name:1:lbw=1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0>
```

---

## Functional Presets Reference

| Preset | Target Functionality | Structural Responsibility |
| :--- | :--- | :--- |
| **`COMPOSITION`** | Overall pose, framing, perspective, spatial layout | Early downsampling blocks in U-Net; early double-stream DiT cross-attention blocks. |
| **`FACE`** | Character likeness, facial structure, anatomical features | Bottleneck layers (`M00`, middle DiT blocks). |
| **`STYLE`** | Color grading, artistic medium, shading, aesthetic tone | Mid-to-late output blocks in U-Net; late single-stream DiT representation blocks. |
| **`TEXTURE`** | Micro-details, line work, high-frequency fine grain | Final terminal output layers. |
| **`RESET`** | Full uniform activation | Sets all block multipliers to `1.0`. |

---

## Cross-Architecture Adaptation & Fail-Proofing

- **Cross-Preset Fallback:** Applying an SDXL preset (e.g. `FACE`) to a Flux or Anima LoRA automatically adapts the weights across the destination block count using linear interpolation.
- **Automatic Vector Expansion:** Legacy 12-block (SDXL) and 17-block (SD 1.5) vectors are expanded to full 26-block U-Net coordinates automatically.
- **Flux Dimension Mapping:** Flux.1 57-block presets seamlessly translate to Flux.2-Klein 9B (32) and 4B (25) by mapping double-stream and single-stream components proportionately.

---

## Installation

1. Open your **Stable Diffusion WebUI Forge** interface.
2. Go to the **Extensions** tab -> **Install from URL**.
3. Enter the repository URL:
   ```text
   https://github.com/0x7flumic/lora-block-weight-neo.git
   ```
4. Click **Install**.
5. Switch to the **Installed** tab and click **Apply and restart UI**.

---