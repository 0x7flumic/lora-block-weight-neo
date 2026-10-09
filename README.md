# LoRA Block Weight Neo

Dynamic, per-block weight control for LoRA models in **Stable Diffusion WebUI Forge (Neo)** across **SD 1.5, SDXL, Flux.1, Flux.2-Klein, Anima, Wan 2.1, and Qwen-Image** architectures.

---

## Quick Start

### Installation
1. In WebUI Forge, navigate to **Extensions** → **Install from URL**.
2. Paste the repository URL:
   ```text
   https://github.com/0x7flumic/lora-block-weight-neo.git
   ```
3. Click **Install**, then click **Apply and restart UI** under the **Installed** tab.

### Usage
- **Interactive Popover:** Left-click or position the text cursor inside any `<lora:...>` tag in your prompt. A floating panel appears allowing you to select presets, tune block sliders, adjust the LoRA weight multiplier, and save custom presets.
- **Inline Syntax:** Add `lbw=PRESET` or a custom vector directly to any LoRA tag:
  ```text
  <lora:my_lora:1.0:lbw=FACE>
  <lora:my_lora:0.8:lbw=STYLE>
  <lora:my_lora:1.0:lbw=1,1,0,0,0,0,1,1,1,0,0,0...>
  ```

---

## Supported Architectures

| Architecture | Block Count | Layer Configuration |
| :--- | :---: | :--- |
| **SD 1.5 & SDXL** | **26** | `BASE` (0), `IN00`–`IN11` (1–12), `M00` (13), `OUT00`–`OUT11` (14–25)<br>*(Auto-expands legacy 12- and 17-block vectors)* |
| **Flux.1 (Dev / Schnell)** | **57** | Double blocks `D00`–`D18` (0–18) + Single blocks `S00`–`S37` (19–56) |
| **Flux.2-Klein 9B** | **32** | Double blocks `D00`–`D07` (0–7) + Single blocks `S00`–`S23` (8–31) |
| **Flux.2-Klein 4B** | **25** | Double blocks `D00`–`D04` (0–4) + Single blocks `S00`–`S19` (5–24) |
| **Anima 2B** | **28** | DiT stack `B00`–`B27` *(LLM adapter protected)* |
| **Anima 2.9B** | **40** | DiT stack `B00`–`B39` *(LLM adapter protected; 28→40 canonical host mapping)* |
| **Anima 3.8B** | **52** | DiT stack `B00`–`B51` *(LLM adapter protected; 28→52 canonical host mapping)* |
| **Wan 2.1 14B / 1.3B** | **40 / 30** | Video DiT stack `B00`–`B39` (14B) / `B00`–`B29` (1.3B) |
| **Qwen-Image** | **60** | Transformer DiT stack `B00`–`B59` |

---

## Built-in Presets

| Preset | Target Functionality | Structural Focus |
| :--- | :--- | :--- |
| **`COMPOSITION`** | Pose, framing, camera angle, spatial structure | Early downsampling U-Net blocks / early DiT cross-attention |
| **`FACE`** | Character likeness, facial features, anatomy | Bottleneck layers (`M00`, middle DiT blocks) |
| **`STYLE`** | Art style, color grading, aesthetics, shading | Late U-Net output blocks / single-stream DiT blocks |
| **`TEXTURE`** | Micro-details, fine line work, grain | Final output layers |
| **`RESET`** | Full uniform activation | Sets all block multipliers to `1.0` |

*(Legacy aliases `MIDD`, `MID`, `INS`, and `OUTS` are supported for backward compatibility.)*

---

## Syntax Guide

### 1. Named Presets (Recommended)
```text
<lora:character_model:1.0:lbw=FACE>
<lora:art_style:0.75:lbw=STYLE>
```

### 2. Custom User Presets
Custom presets saved via the popover can be referenced case-insensitively:
```text
<lora:character_model:1.0:lbw=MyCustomPreset>
```

### 3. Comma-Separated Vectors
Supply exact float multipliers per block:
```text
<lora:my_lora:1.0:lbw=1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0>
```

### 4. Single Block Isolation
Target a single layer (one-hot activation):
```text
<lora:my_lora:1.0:lbw=M00>
<lora:anima_lora:1.0:lbw=B14>
```