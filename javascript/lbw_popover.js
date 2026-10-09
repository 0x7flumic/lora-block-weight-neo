/**
 * LoRA Block Weight Neo - Interactive In-Prompt Popover
 *
 * Appears when the user left-clicks or places the caret inside a <lora:NAME:...> token.
 * Provides:
 *  - Automatic LoRA architecture detection from Extra Networks metadata
 *  - Architecture switcher with SD/XL, Flux.1, Klein 9B/4B, Anima (28/40/52), Wan, Qwen-Image
 *  - Quick-select functional preset badges without resetting LoRA weight
 *  - Interactive block chips / custom block weight editor for all blocks
 *  - Real-time in-prompt synchronization with Forge token counters
 */

(function () {
    "use strict";

    let popoverElem = null;
    let activeTextarea = null;
    let currentLoraMatch = null;
    let activeArch = "sdxl";
    let manualOverrideLora = null;
    const loraArchCache = {};

    // Architecture definitions with block counts and block labels
    const ARCH_DEFS = {
        sdxl: {
            label: "SD / SDXL",
            count: 26,
            names: ["BASE", ...Array.from({length: 12}, (_, i) => `IN${String(i).padStart(2, '0')}`), "M00", ...Array.from({length: 12}, (_, i) => `OUT${String(i).padStart(2, '0')}`)],
            presets: {
                COMPOSITION: "1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
                FACE:        "0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0",
                STYLE:       "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1",
                TEXTURE:     "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1"
            }
        },
        flux1: {
            label: "Flux.1",
            count: 57,
            names: [...Array.from({length: 19}, (_, i) => `D${String(i).padStart(2, '0')}`), ...Array.from({length: 38}, (_, i) => `S${String(i).padStart(2, '0')}`)],
            presets: {
                COMPOSITION: [...Array(10).fill("1"), ...Array(9).fill("0"), ...Array(38).fill("0")].join(","),
                FACE:        [...Array(10).fill("0"), ...Array(9).fill("1"), ...Array(10).fill("1"), ...Array(28).fill("0")].join(","),
                STYLE:       [...Array(19).fill("0"), ...Array(10).fill("0"), ...Array(28).fill("1")].join(","),
                TEXTURE:     [...Array(19).fill("0"), ...Array(28).fill("0"), ...Array(10).fill("1")].join(",")
            }
        },
        flux_k9b: {
            label: "Flux.2-Klein 9B",
            count: 32,
            names: [...Array.from({length: 8}, (_, i) => `D${String(i).padStart(2, '0')}`), ...Array.from({length: 24}, (_, i) => `S${String(i).padStart(2, '0')}`)],
            presets: {
                COMPOSITION: [...Array(4).fill("1"), ...Array(4).fill("0"), ...Array(24).fill("0")].join(","),
                FACE:        [...Array(4).fill("0"), ...Array(4).fill("1"), ...Array(6).fill("1"), ...Array(18).fill("0")].join(","),
                STYLE:       [...Array(8).fill("0"), ...Array(6).fill("0"), ...Array(18).fill("1")].join(","),
                TEXTURE:     [...Array(8).fill("0"), ...Array(18).fill("0"), ...Array(6).fill("1")].join(",")
            }
        },
        flux_k4b: {
            label: "Flux.2-Klein 4B",
            count: 25,
            names: [...Array.from({length: 5}, (_, i) => `D${String(i).padStart(2, '0')}`), ...Array.from({length: 20}, (_, i) => `S${String(i).padStart(2, '0')}`)],
            presets: {
                COMPOSITION: [...Array(3).fill("1"), ...Array(2).fill("0"), ...Array(20).fill("0")].join(","),
                FACE:        [...Array(3).fill("0"), ...Array(2).fill("1"), ...Array(5).fill("1"), ...Array(15).fill("0")].join(","),
                STYLE:       [...Array(5).fill("0"), ...Array(5).fill("0"), ...Array(15).fill("1")].join(","),
                TEXTURE:     [...Array(5).fill("0"), ...Array(15).fill("0"), ...Array(5).fill("1")].join(",")
            }
        },
        anima28: {
            label: "Anima 2B (28)",
            count: 28,
            names: Array.from({length: 28}, (_, i) => `B${String(i).padStart(2, '0')}`),
            presets: {
                COMPOSITION: [...Array(7).fill("1"), ...Array(21).fill("0")].join(","),
                FACE:        [...Array(7).fill("0"), ...Array(11).fill("1"), ...Array(10).fill("0")].join(","),
                STYLE:       [...Array(14).fill("0"), ...Array(14).fill("1")].join(","),
                TEXTURE:     [...Array(24).fill("0"), ...Array(4).fill("1")].join(",")
            }
        },
        anima40: {
            label: "Anima 2.9B (40)",
            count: 40,
            names: Array.from({length: 40}, (_, i) => `B${String(i).padStart(2, '0')}`),
            presets: (() => {
                const map28_40 = [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 22, 22, 23, 24, 24, 25, 26, 27];
                const base = {
                    COMPOSITION: [...Array(7).fill("1"), ...Array(21).fill("0")],
                    FACE:        [...Array(7).fill("0"), ...Array(11).fill("1"), ...Array(10).fill("0")],
                    STYLE:       [...Array(14).fill("0"), ...Array(14).fill("1")],
                    TEXTURE:     [...Array(24).fill("0"), ...Array(4).fill("1")]
                };
                const out = {};
                for (const [k, arr] of Object.entries(base)) {
                    out[k] = map28_40.map(idx => arr[idx]).join(",");
                }
                return out;
            })()
        },
        anima52: {
            label: "Anima 3.8B (52)",
            count: 52,
            names: Array.from({length: 52}, (_, i) => `B${String(i).padStart(2, '0')}`),
            presets: (() => {
                const map28_52 = [0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7, 8, 9, 9, 9, 10, 11, 11, 11, 12, 13, 14, 14, 14, 15, 16, 16, 16, 17, 18, 18, 18, 19, 20, 20, 20, 21, 22, 22, 22, 23, 24, 24, 24, 25, 26, 27];
                const base = {
                    COMPOSITION: [...Array(7).fill("1"), ...Array(21).fill("0")],
                    FACE:        [...Array(7).fill("0"), ...Array(11).fill("1"), ...Array(10).fill("0")],
                    STYLE:       [...Array(14).fill("0"), ...Array(14).fill("1")],
                    TEXTURE:     [...Array(24).fill("0"), ...Array(4).fill("1")]
                };
                const out = {};
                for (const [k, arr] of Object.entries(base)) {
                    out[k] = map28_52.map(idx => arr[idx]).join(",");
                }
                return out;
            })()
        },
        wan21_14b: {
            label: "Wan 2.1-14B (40)",
            count: 40,
            names: Array.from({length: 40}, (_, i) => `B${String(i).padStart(2, '0')}`),
            presets: {
                COMPOSITION: [...Array(10).fill("1"), ...Array(30).fill("0")].join(","),
                FACE:        [...Array(10).fill("0"), ...Array(15).fill("1"), ...Array(15).fill("0")].join(","),
                STYLE:       [...Array(20).fill("0"), ...Array(20).fill("1")].join(","),
                TEXTURE:     [...Array(32).fill("0"), ...Array(8).fill("1")].join(",")
            }
        },
        wan21_13b: {
            label: "Wan 2.1-1.3B (30)",
            count: 30,
            names: Array.from({length: 30}, (_, i) => `B${String(i).padStart(2, '0')}`),
            presets: {
                COMPOSITION: [...Array(8).fill("1"), ...Array(22).fill("0")].join(","),
                FACE:        [...Array(8).fill("0"), ...Array(12).fill("1"), ...Array(10).fill("0")].join(","),
                STYLE:       [...Array(15).fill("0"), ...Array(15).fill("1")].join(","),
                TEXTURE:     [...Array(24).fill("0"), ...Array(6).fill("1")].join(",")
            }
        },
        qwen_image: {
            label: "Qwen-Image (60)",
            count: 60,
            names: Array.from({length: 60}, (_, i) => `B${String(i).padStart(2, '0')}`),
            presets: {
                COMPOSITION: [...Array(15).fill("1"), ...Array(45).fill("0")].join(","),
                FACE:        [...Array(15).fill("0"), ...Array(25).fill("1"), ...Array(20).fill("0")].join(","),
                STYLE:       [...Array(30).fill("0"), ...Array(30).fill("1")].join(","),
                TEXTURE:     [...Array(50).fill("0"), ...Array(10).fill("1")].join(",")
            }
        }
    };

    const PRESET_BADGES = [
        { name: "COMPOSITION", desc: "Pose & Spatial Layout", color: "#3b82f6" },
        { name: "FACE", desc: "Identity & Anatomical Features", color: "#10b981" },
        { name: "STYLE", desc: "Aesthetics & Color Palette", color: "#8b5cf6" },
        { name: "TEXTURE", desc: "Fine Grain & Micro Details", color: "#ec4899" },
        { name: "RESET", desc: "Remove LBW Filter", color: "#6b7280" }
    ];

    /**
     * Inspect Extra Networks metadata or base model checkpoint to determine architecture.
     */
    async function resolveArchitectureForLora(loraName) {
        if (!loraName) return "sdxl";

        // 1. In-memory session cache lookup (0ms)
        if (loraArchCache[loraName]) {
            return loraArchCache[loraName];
        }

        // 2. Direct safetensors header inspection via backend API (<1ms)
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 1500);
            const res = await fetch(`./lbw/api/detect_lora?name=${encodeURIComponent(loraName)}`, {
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            if (res.ok) {
                const data = await res.json();
                if (data && data.arch && ARCH_DEFS[data.arch]) {
                    loraArchCache[loraName] = data.arch;
                    return data.arch;
                }
            }
        } catch (e) {}

        // 3. Fallback: Check Extra Networks card metadata cache or endpoint
        try {
            const card = gradioApp().querySelector(`.extra-network-cards .card[data-name="${CSS.escape(loraName)}"]`);
            if (card) {
                const cardText = (card.innerText || "").toLowerCase();
                let detected = null;
                if (cardText.includes("anima 52") || cardText.includes("anima-52") || cardText.includes("anima 3.8b") || cardText.includes("anima-3.8b")) detected = "anima52";
                else if (cardText.includes("anima 2.9b") || cardText.includes("anima-2.9b") || cardText.includes("anima 40")) detected = "anima40";
                else if (cardText.includes("anima 2b") || cardText.includes("anima-2b") || cardText.includes("anima 28")) detected = "anima28";
                else if (cardText.includes("anima")) detected = "anima28";
                else if (cardText.includes("flux 9b") || cardText.includes("klein 9b")) detected = "flux_k9b";
                else if (cardText.includes("flux 4b") || cardText.includes("klein 4b")) detected = "flux_k4b";
                else if (cardText.includes("flux")) detected = "flux1";
                else if (cardText.includes("wan") && (cardText.includes("1.3b") || cardText.includes("13b") || cardText.includes("30"))) detected = "wan21_13b";
                else if (cardText.includes("wan")) detected = "wan21_14b";
                else if (cardText.includes("qwen")) detected = "qwen_image";
                else if (cardText.includes("sdxl") || cardText.includes("pony") || cardText.includes("illustrious")) detected = "sdxl";

                if (detected) {
                    loraArchCache[loraName] = detected;
                    return detected;
                }
            }

            // Query backend metadata endpoint with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 1500);
            const res = await fetch(`./sd_extra_networks/metadata?page=lora&item=${encodeURIComponent(loraName)}`, {
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            if (res.ok) {
                const data = await res.json();
                const metaStr = (typeof data.metadata === "string" ? data.metadata : JSON.stringify(data.metadata || "")).toLowerCase();
                let detected = null;
                if (metaStr.includes("anima 52") || metaStr.includes("anima-52") || metaStr.includes("anima 3.8b") || metaStr.includes("anima-3.8b") || metaStr.includes("blocks.51")) detected = "anima52";
                else if (metaStr.includes("anima 2.9b") || metaStr.includes("anima-2.9b") || metaStr.includes("anima2.9b") || metaStr.includes("blocks.39")) detected = "anima40";
                else if (metaStr.includes("anima 2b") || metaStr.includes("anima-2b") || metaStr.includes("anima2b")) detected = "anima28";
                else if (metaStr.includes("anima")) {
                    detected = (metaStr.includes("blocks.5") || metaStr.includes("blocks.4")) ? "anima52" : ((metaStr.includes("blocks.3") || metaStr.includes("blocks.28")) ? "anima40" : "anima28");
                } else if (metaStr.includes("flux") && (metaStr.includes("9b") || metaStr.includes("klein"))) detected = "flux_k9b";
                else if (metaStr.includes("flux") && metaStr.includes("4b")) detected = "flux_k4b";
                else if (metaStr.includes("flux")) detected = "flux1";
                else if (metaStr.includes("wan")) {
                    detected = (metaStr.includes("1.3b") || metaStr.includes("13b")) ? "wan21_13b" : "wan21_14b";
                } else if (metaStr.includes("qwen")) detected = "qwen_image";
                else if (metaStr.includes("sdxl") || metaStr.includes("sd_xl") || metaStr.includes("v1-5") || metaStr.includes("sd15")) detected = "sdxl";

                if (detected) {
                    loraArchCache[loraName] = detected;
                    return detected;
                }
            }
        } catch (e) {}

        // 4. Fallback to active checkpoint
        try {
            const ckptSelect = gradioApp().querySelector("#setting_sd_model_checkpoint select, #sd_checkpoint_hash");
            let ckptText = "";
            if (ckptSelect) {
                ckptText = (ckptSelect.value || ckptSelect.textContent || "").toLowerCase();
            }
            let detected = null;
            if (ckptText.includes("anima 52") || ckptText.includes("anima 3.8b") || ckptText.includes("anima-3.8b")) detected = "anima52";
            else if (ckptText.includes("anima 2.9b") || ckptText.includes("anima-2.9b") || ckptText.includes("anima 40")) detected = "anima40";
            else if (ckptText.includes("anima")) detected = "anima28";
            else if (ckptText.includes("flux") && (ckptText.includes("klein") || ckptText.includes("9b"))) detected = "flux_k9b";
            else if (ckptText.includes("flux") && ckptText.includes("4b")) detected = "flux_k4b";
            else if (ckptText.includes("flux")) detected = "flux1";
            else if (ckptText.includes("wan") && (ckptText.includes("1.3b") || ckptText.includes("13b"))) detected = "wan21_13b";
            else if (ckptText.includes("wan")) detected = "wan21_14b";
            else if (ckptText.includes("qwen")) detected = "qwen_image";
            else if (ckptText.includes("xl") || ckptText.includes("sdxl")) detected = "sdxl";

            if (detected) {
                loraArchCache[loraName] = detected;
                return detected;
            }
        } catch (e) {}

        return "sdxl";
    }

    let customPresets = {}; // arch -> { name: weights_str }

    // Fetch custom presets from FastAPI backend
    async function loadPresetsFromServer() {
        try {
            const res = await fetch("./lbw/api/presets");
            if (res.ok) {
                const data = await res.json();
                if (data && data.presets) {
                    customPresets = data.presets;
                    renderPresetBadges();
                }
            }
        } catch (e) {
            console.debug("[LBW] Could not load presets from server:", e);
        }
    }

    /**
     * Create or retrieve the popover DOM element.
     */
    function getOrCreatePopover() {
        if (popoverElem) return popoverElem;

        popoverElem = document.createElement("div");
        popoverElem.id = "lbw-inprompt-popover";
        popoverElem.className = "lbw-popover-container";
        popoverElem.style.display = "none";

        popoverElem.innerHTML = `
            <div class="lbw-popover-header">
                <div class="lbw-popover-title">
                    <span class="lbw-popover-icon">⚡</span>
                    <span class="lbw-popover-lora-name" id="lbw-popover-name">LoRA</span>
                </div>
                <div class="lbw-popover-arch-wrap">
                    <select id="lbw-popover-arch-select" class="lbw-arch-select" title="Target Architecture">
                        ${Object.keys(ARCH_DEFS).map(id => `<option value="${id}">${ARCH_DEFS[id].label}</option>`).join("")}
                    </select>
                    <button id="lbw-popover-close" class="lbw-close-btn" title="Close (Esc)">✕</button>
                </div>
            </div>
            <div class="lbw-popover-body">
                <div class="lbw-presets-header-row">
                    <span class="lbw-presets-heading">Presets</span>
                    <button type="button" id="lbw-manage-presets-toggle" class="lbw-manage-toggle-btn" title="Manage Custom Presets">Manage ⚙</button>
                </div>
                <div class="lbw-popover-badges" id="lbw-badges-list">
                    <!-- Badges rendered dynamically -->
                </div>
                <div class="lbw-manage-presets-panel" id="lbw-manage-presets-panel" style="display: none;">
                    <div class="lbw-manage-presets-list" id="lbw-manage-presets-list">
                        <!-- Custom presets list with delete buttons rendered here -->
                    </div>
                </div>
                <div class="lbw-popover-slider-row">
                    <span class="lbw-slider-label">LoRA Weight:</span>
                    <input type="range" id="lbw-weight-slider" min="0" max="2" step="0.01" value="1.0" class="lbw-range">
                    <input type="number" id="lbw-weight-input" min="-5" max="5" step="0.01" value="1.0" class="lbw-slider-num-input">
                </div>
                <div class="lbw-blocks-section">
                    <div class="lbw-blocks-header-row">
                        <span class="lbw-blocks-heading">Block Weights (<span id="lbw-blocks-count-label">26</span> blocks)</span>
                        <div class="lbw-blocks-quick-actions">
                            <button type="button" id="lbw-blocks-all-one" class="lbw-blocks-subbtn">All 1.0</button>
                            <button type="button" id="lbw-blocks-all-zero" class="lbw-blocks-subbtn">All 0.0</button>
                            <button type="button" id="lbw-blocks-invert" class="lbw-blocks-subbtn">Invert</button>
                        </div>
                    </div>
                    <div id="lbw-blocks-grid" class="lbw-blocks-grid"></div>
                </div>
                <div class="lbw-save-preset-bar">
                    <input type="text" id="lbw-preset-name-input" class="lbw-preset-input" placeholder="Preset name (e.g. MyPreset)">
                    <button type="button" id="lbw-save-preset-btn" class="lbw-save-btn">Save Preset</button>
                </div>
            </div>
        `;

        document.body.appendChild(popoverElem);

        // Bind popover header controls
        const closeBtn = popoverElem.querySelector("#lbw-popover-close");
        closeBtn.addEventListener("click", hidePopover);

        const archSelect = popoverElem.querySelector("#lbw-popover-arch-select");
        archSelect.addEventListener("change", (e) => {
            activeArch = e.target.value;
            manualOverrideLora = currentLoraMatch ? currentLoraMatch.name : null;
            renderPresetBadges();
            renderBlockSliders();
            if (managePanel && managePanel.style.display !== "none") {
                renderManagePresetsPanel();
            }
        });

        // Manage Presets Toggle Button
        const manageToggleBtn = popoverElem.querySelector("#lbw-manage-presets-toggle");
        const managePanel = popoverElem.querySelector("#lbw-manage-presets-panel");
        manageToggleBtn.addEventListener("click", () => {
            const isHidden = managePanel.style.display === "none";
            managePanel.style.display = isHidden ? "block" : "none";
            manageToggleBtn.classList.toggle("active", isHidden);
            if (isHidden) {
                renderManagePresetsPanel();
            }
        });

        // LoRA Multiplier Controls (slider + numeric input)
        const slider = popoverElem.querySelector("#lbw-weight-slider");
        const numInput = popoverElem.querySelector("#lbw-weight-input");

        slider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            numInput.value = isNaN(val) ? 1.0 : val;
            updateLoraWeight(val);
        });

        numInput.addEventListener("change", (e) => {
            const val = parseFloat(e.target.value);
            const safeVal = isNaN(val) ? 1.0 : val;
            slider.value = safeVal;
            updateLoraWeight(safeVal);
        });

        // Block Quick Action Buttons
        popoverElem.querySelector("#lbw-blocks-all-one").addEventListener("click", () => setAllBlocks(1.0));
        popoverElem.querySelector("#lbw-blocks-all-zero").addEventListener("click", () => setAllBlocks(0.0));
        popoverElem.querySelector("#lbw-blocks-invert").addEventListener("click", invertBlocks);

        // Save Custom Preset Handler
        const saveBtn = popoverElem.querySelector("#lbw-save-preset-btn");
        const nameInput = popoverElem.querySelector("#lbw-preset-name-input");
        saveBtn.addEventListener("click", async () => {
            const name = nameInput.value.trim();
            if (!name) return;
            const weights = getCurrentBlockWeightsArray().join(",");
            saveBtn.disabled = true;
            saveBtn.textContent = "Saving...";
            try {
                const res = await fetch("./lbw/api/presets", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        arch: activeArch,
                        name: name,
                        weights: weights
                    })
                });
                if (res.ok) {
                    const data = await res.json();
                    if (data && data.presets) {
                        customPresets = data.presets;
                    }
                    nameInput.value = "";
                    renderPresetBadges();
                    renderManagePresetsPanel();
                    saveBtn.textContent = "Saved!";
                    setTimeout(() => {
                        saveBtn.textContent = "Save Preset";
                        saveBtn.disabled = false;
                    }, 1200);
                } else {
                    saveBtn.textContent = "Error";
                    setTimeout(() => {
                        saveBtn.textContent = "Save Preset";
                        saveBtn.disabled = false;
                    }, 1500);
                }
            } catch (err) {
                console.error("[LBW] Save preset failed:", err);
                saveBtn.textContent = "Error";
                setTimeout(() => {
                    saveBtn.textContent = "Save Preset";
                    saveBtn.disabled = false;
                }, 1500);
            }
        });

        renderPresetBadges();
        loadPresetsFromServer();

        return popoverElem;
    }

    /**
     * Delete a custom preset from the server and local state.
     */
    async function deleteCustomPreset(presetName) {
        try {
            const res = await fetch("./lbw/api/presets", {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    arch: activeArch,
                    name: presetName
                })
            });
            if (res.ok) {
                const data = await res.json();
                if (data && data.presets) {
                    customPresets = data.presets;
                } else if (customPresets[activeArch]) {
                    delete customPresets[activeArch][presetName];
                }
                renderPresetBadges();
                renderManagePresetsPanel();
            }
        } catch (err) {
            console.error("[LBW] Delete preset failed:", err);
        }
    }

    /**
     * Render the inline Manage Presets panel for the currently selected architecture.
     */
    function renderManagePresetsPanel() {
        if (!popoverElem) return;
        const panel = popoverElem.querySelector("#lbw-manage-presets-panel");
        const list = popoverElem.querySelector("#lbw-manage-presets-list");
        if (!panel || !list) return;

        const archPresets = customPresets[activeArch] || {};
        const presetNames = Object.keys(archPresets);

        list.innerHTML = "";

        if (presetNames.length === 0) {
            list.innerHTML = `<div class="lbw-manage-empty">No custom presets for ${ARCH_DEFS[activeArch] ? ARCH_DEFS[activeArch].label : activeArch}.</div>`;
            return;
        }

        presetNames.forEach(name => {
            const row = document.createElement("div");
            row.className = "lbw-manage-preset-row";

            const nameSpan = document.createElement("span");
            nameSpan.className = "lbw-manage-preset-name";
            nameSpan.textContent = name;
            nameSpan.title = name;

            const weightsSpan = document.createElement("span");
            weightsSpan.className = "lbw-manage-preset-weights";
            const wVal = archPresets[name] || "";
            weightsSpan.textContent = wVal.length > 32 ? wVal.substring(0, 30) + "..." : wVal;
            weightsSpan.title = wVal;

            const delBtn = document.createElement("button");
            delBtn.type = "button";
            delBtn.className = "lbw-manage-delete-btn";
            delBtn.innerHTML = "✕ Delete";
            delBtn.title = `Delete preset "${name}"`;
            delBtn.addEventListener("click", () => deleteCustomPreset(name));

            row.appendChild(nameSpan);
            row.appendChild(weightsSpan);
            row.appendChild(delBtn);
            list.appendChild(row);
        });
    }

    /**
     * Render preset badges dynamically for active architecture, including built-ins and server presets.
     */
    function renderPresetBadges() {
        if (!popoverElem) return;
        const container = popoverElem.querySelector("#lbw-badges-list");
        if (!container) return;
        container.innerHTML = "";

        const arch = ARCH_DEFS[activeArch] || ARCH_DEFS["sdxl"];

        // 1. Built-in badges
        PRESET_BADGES.forEach(p => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "lbw-badge-btn";
            btn.setAttribute("data-preset", p.name);
            btn.style.setProperty("--badge-color", p.color);
            btn.title = p.desc;
            btn.innerHTML = `<span class="lbw-badge-dot"></span>${p.name}`;
            btn.addEventListener("click", () => applyPreset(p.name));
            container.appendChild(btn);
        });

        // 2. Custom saved presets from server
        const archPresets = customPresets[activeArch] || {};
        Object.keys(archPresets).forEach(customName => {
            // Avoid duplicating built-in names
            if (PRESET_BADGES.some(b => b.name.toUpperCase() === customName.toUpperCase())) return;
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "lbw-badge-btn lbw-custom-badge";
            btn.setAttribute("data-preset", customName);
            // Distinct orange color matching the style of built-in presets
            btn.style.setProperty("--badge-color", "#f59e0b");
            btn.title = `Custom: ${customName}`;
            const dot = document.createElement("span");
            dot.className = "lbw-badge-dot";
            btn.appendChild(dot);
            btn.appendChild(document.createTextNode(customName));
            btn.addEventListener("click", () => applyPreset(customName, archPresets[customName]));
            container.appendChild(btn);
        });

        updatePresetBadgeHighlights();
    }

    /**
     * Highlight the active preset badge if current block weight vector matches.
     */
    function updatePresetBadgeHighlights() {
        if (!popoverElem || !currentLoraMatch) return;
        const badges = popoverElem.querySelectorAll(".lbw-badge-btn");
        const currentLbw = (currentLoraMatch.lbw || "").trim();

        badges.forEach(b => {
            const pName = b.getAttribute("data-preset");
            if (pName === "RESET") {
                b.classList.toggle("active", !currentLbw);
                return;
            }

            const arch = ARCH_DEFS[activeArch] || ARCH_DEFS["sdxl"];
            const vector = (customPresets[activeArch] && customPresets[activeArch][pName]) || arch.presets[pName];
            if (vector && currentLbw === vector) {
                b.classList.add("active");
            } else {
                b.classList.remove("active");
            }
        });
    }

    /**
     * Parse currently active block weights into a numeric array matching the active architecture count.
     */
    function getCurrentBlockWeightsArray() {
        const arch = ARCH_DEFS[activeArch] || ARCH_DEFS["sdxl"];
        const targetCount = arch.count;

        if (!currentLoraMatch || !currentLoraMatch.lbw) {
            return Array(targetCount).fill(1.0);
        }

        const lbwVal = currentLoraMatch.lbw.trim();

        // Check if custom or built-in preset name in active arch
        const archPresets = customPresets[activeArch] || {};
        if (archPresets[lbwVal]) {
            return archPresets[lbwVal].split(",").map(v => parseFloat(v.trim()) || 0.0);
        }
        if (arch.presets[lbwVal.toUpperCase()]) {
            return arch.presets[lbwVal.toUpperCase()].split(",").map(v => parseFloat(v.trim()) || 0.0);
        }

        // Check if comma-separated list
        if (lbwVal.includes(",")) {
            const parts = lbwVal.split(",").map(v => parseFloat(v.trim()) || 0.0);
            if (parts.length === targetCount) return parts;
            if (parts.length < targetCount) return [...parts, ...Array(targetCount - parts.length).fill(1.0)];
            return parts.slice(0, targetCount);
        }

        return Array(targetCount).fill(1.0);
    }

    /**
     * Render the multi-column grid of block sliders and numeric inputs.
     */
    function renderBlockSliders() {
        const grid = popoverElem.querySelector("#lbw-blocks-grid");
        const countLabel = popoverElem.querySelector("#lbw-blocks-count-label");
        const arch = ARCH_DEFS[activeArch] || ARCH_DEFS["sdxl"];

        countLabel.textContent = arch.count;
        grid.innerHTML = "";

        const weights = getCurrentBlockWeightsArray();

        arch.names.forEach((name, i) => {
            const val = weights[i] !== undefined ? weights[i] : 1.0;
            const card = document.createElement("div");
            card.className = "lbw-block-slider-card";
            card.innerHTML = `
                <div class="lbw-card-header">
                    <span class="lbw-card-block-name">${name}</span>
                    <input type="number" class="lbw-card-num-input" min="-5" max="5" step="0.01" value="${val}" data-index="${i}">
                </div>
                <input type="range" class="lbw-card-range-slider" min="0" max="2" step="0.01" value="${Math.max(0, Math.min(2, val))}" data-index="${i}">
            `;

            const rangeInput = card.querySelector(".lbw-card-range-slider");
            const numInput = card.querySelector(".lbw-card-num-input");

            rangeInput.addEventListener("input", (e) => {
                const newVal = parseFloat(e.target.value);
                numInput.value = isNaN(newVal) ? 1.0 : newVal;
                commitBlockWeightsFromSliders();
            });

            numInput.addEventListener("change", (e) => {
                const newVal = parseFloat(e.target.value);
                const safeVal = isNaN(newVal) ? 1.0 : newVal;
                rangeInput.value = Math.max(0, Math.min(2, safeVal));
                commitBlockWeightsFromSliders();
            });

            grid.appendChild(card);
        });
    }

    /**
     * Read block slider values and rewrite prompt tag.
     */
    function commitBlockWeightsFromSliders() {
        if (!activeTextarea || !currentLoraMatch) return;

        const numInputs = popoverElem.querySelectorAll("#lbw-blocks-grid .lbw-card-num-input");
        const vals = Array.from(numInputs).map(inp => {
            const v = parseFloat(inp.value);
            return isNaN(v) ? 0.0 : Math.round(v * 1000) / 1000;
        });
        const vectorStr = vals.join(",");

        const loraWeight = currentLoraMatch.weight !== null ? currentLoraMatch.weight : 1.0;
        const newTag = `<lora:${currentLoraMatch.name}:${loraWeight}:lbw=${vectorStr}>`;

        const val = activeTextarea.value;
        const before = val.substring(0, currentLoraMatch.startIdx);
        const after = val.substring(currentLoraMatch.endIdx);

        activeTextarea.value = before + newTag + after;

        const newEndIdx = currentLoraMatch.startIdx + newTag.length;
        currentLoraMatch.fullTag = newTag;
        currentLoraMatch.endIdx = newEndIdx;
        currentLoraMatch.lbw = vectorStr;

        if (typeof updateInput === "function") {
            updateInput(activeTextarea);
        } else {
            activeTextarea.dispatchEvent(new Event("input", { bubbles: true }));
        }

        updatePresetBadgeHighlights();
    }

    function setAllBlocks(val) {
        const numInputs = popoverElem.querySelectorAll("#lbw-blocks-grid .lbw-card-num-input");
        const rangeInputs = popoverElem.querySelectorAll("#lbw-blocks-grid .lbw-card-range-slider");
        numInputs.forEach((inp, i) => {
            inp.value = val;
            if (rangeInputs[i]) rangeInputs[i].value = Math.max(0, Math.min(2, val));
        });
        commitBlockWeightsFromSliders();
    }

    function invertBlocks() {
        const numInputs = popoverElem.querySelectorAll("#lbw-blocks-grid .lbw-card-num-input");
        const rangeInputs = popoverElem.querySelectorAll("#lbw-blocks-grid .lbw-card-range-slider");
        numInputs.forEach((inp, i) => {
            const cur = parseFloat(inp.value) || 0.0;
            const inv = Math.round((1.0 - cur) * 100) / 100;
            inp.value = inv;
            if (rangeInputs[i]) rangeInputs[i].value = Math.max(0, Math.min(2, inv));
        });
        commitBlockWeightsFromSliders();
    }

    /**
     * Show and position the popover above the active cursor or textarea.
     */
    function showPopover(match, textarea) {
        activeTextarea = textarea;
        currentLoraMatch = match;

        const pop = getOrCreatePopover();
        const nameElem = pop.querySelector("#lbw-popover-name");
        const archSelect = pop.querySelector("#lbw-popover-arch-select");
        const slider = pop.querySelector("#lbw-weight-slider");
        const numInput = pop.querySelector("#lbw-weight-input");

        nameElem.textContent = match.name.length > 28 ? match.name.substring(0, 25) + "..." : match.name;
        nameElem.title = match.name;

        // Synchronously initialize architecture with current or fallback
        if (!ARCH_DEFS[activeArch]) {
            activeArch = "sdxl";
        }
        archSelect.value = activeArch;

        slider.value = match.weight;
        numInput.value = parseFloat(match.weight);

        // Immediate render so popover is visible without waiting for network
        renderPresetBadges();
        renderBlockSliders();
        updatePresetBadgeHighlights();

        const rect = textarea.getBoundingClientRect();
        pop.style.visibility = "hidden";
        pop.style.display = "block";

        const popWidth = Math.min(780, window.innerWidth * 0.96);
        const popHeight = pop.offsetHeight || 380;

        let left = rect.left + window.scrollX + 10;
        // Position below the textarea if space allows, otherwise above
        let top = rect.bottom + window.scrollY + 8;
        if (top + popHeight > window.scrollY + window.innerHeight - 10 && rect.top - popHeight > 10) {
            top = rect.top + window.scrollY - popHeight - 8;
        }

        if (left + popWidth > window.innerWidth - 10) {
            left = window.innerWidth - popWidth - 10;
        }
        if (top < 10) {
            top = 10;
        }

        pop.style.left = Math.max(10, left) + "px";
        pop.style.top = Math.max(10, top) + "px";
        pop.style.visibility = "visible";

        // Asynchronously detect architecture from safetensors header or metadata in background
        const requestedLora = match.name;
        if (manualOverrideLora && manualOverrideLora !== requestedLora) {
            manualOverrideLora = null;
        }

        resolveArchitectureForLora(requestedLora).then(detectedArch => {
            if (manualOverrideLora === requestedLora) {
                return; // Respect manual user override for this editing session
            }
            if (currentLoraMatch && currentLoraMatch.name === requestedLora && detectedArch && detectedArch !== activeArch) {
                activeArch = detectedArch;
                archSelect.value = activeArch;
                renderPresetBadges();
                renderBlockSliders();
                updatePresetBadgeHighlights();
            }
        }).catch(() => {});
    }

    /**
     * Hide the popover.
     */
    function hidePopover() {
        if (popoverElem) {
            popoverElem.style.display = "none";
        }
        currentLoraMatch = null;
    }

    /**
     * Apply preset to prompt in-place, writing the full comma-separated block weight vector
     * and strictly preserving the existing LoRA weight multiplier.
     */
    function applyPreset(presetName, customVectorStr = null) {
        if (!activeTextarea || !currentLoraMatch) return;

        let vectorStr = customVectorStr;
        if (!vectorStr) {
            if (presetName === "RESET") {
                vectorStr = "";
            } else {
                const arch = ARCH_DEFS[activeArch] || ARCH_DEFS["sdxl"];
                vectorStr = (customPresets[activeArch] && customPresets[activeArch][presetName]) || arch.presets[presetName] || "";
            }
        }

        const loraWeight = currentLoraMatch.weight !== null ? currentLoraMatch.weight : 1.0;
        let newTag = "";
        if (!vectorStr) {
            newTag = `<lora:${currentLoraMatch.name}:${loraWeight}>`;
        } else {
            newTag = `<lora:${currentLoraMatch.name}:${loraWeight}:lbw=${vectorStr}>`;
        }

        const val = activeTextarea.value;
        const before = val.substring(0, currentLoraMatch.startIdx);
        const after = val.substring(currentLoraMatch.endIdx);

        activeTextarea.value = before + newTag + after;

        const newEndIdx = currentLoraMatch.startIdx + newTag.length;
        currentLoraMatch.fullTag = newTag;
        currentLoraMatch.endIdx = newEndIdx;
        currentLoraMatch.lbw = vectorStr || null;

        activeTextarea.setSelectionRange(currentLoraMatch.startIdx, newEndIdx);

        if (typeof updateInput === "function") {
            updateInput(activeTextarea);
        } else {
            activeTextarea.dispatchEvent(new Event("input", { bubbles: true }));
        }

        renderBlockSliders();
        updatePresetBadgeHighlights();
    }


    /**
     * Update LoRA strength multiplier in prompt without touching lbw vector.
     */
    function updateLoraWeight(newWeight) {
        if (!activeTextarea || !currentLoraMatch) return;

        const parsedWeight = parseFloat(newWeight);
        if (isNaN(parsedWeight)) return;

        currentLoraMatch.weight = parsedWeight;

        let newTag = "";
        if (currentLoraMatch.lbw) {
            newTag = `<lora:${currentLoraMatch.name}:${parsedWeight}:lbw=${currentLoraMatch.lbw}>`;
        } else {
            newTag = `<lora:${currentLoraMatch.name}:${parsedWeight}>`;
        }

        const val = activeTextarea.value;
        const before = val.substring(0, currentLoraMatch.startIdx);
        const after = val.substring(currentLoraMatch.endIdx);

        activeTextarea.value = before + newTag + after;

        const newEndIdx = currentLoraMatch.startIdx + newTag.length;
        currentLoraMatch.fullTag = newTag;
        currentLoraMatch.endIdx = newEndIdx;

        if (typeof updateInput === "function") {
            updateInput(activeTextarea);
        } else {
            activeTextarea.dispatchEvent(new Event("input", { bubbles: true }));
        }
    }

    /**
     * Look for <lora:NAME:...> surrounding current caret position in textarea.
     */
    function findLoraAtCursor(textarea) {
        if (!textarea) return null;
        const pos = textarea.selectionStart;
        const text = textarea.value;

        const openIdx = text.lastIndexOf("<lora:", pos);
        if (openIdx === -1) return null;

        const closeIdx = text.indexOf(">", openIdx);
        if (closeIdx === -1 || pos > closeIdx + 1) return null;

        const nextOpen = text.indexOf("<", openIdx + 1);
        if (nextOpen !== -1 && nextOpen < closeIdx) return null;

        const tagStr = text.substring(openIdx, closeIdx + 1);
        const content = tagStr.slice(6, -1);
        const parts = content.split(":");
        const name = parts[0] ? parts[0].trim() : "";
        if (!name) return null;

        let weight = null;
        let lbw = null;

        for (let i = 1; i < parts.length; i++) {
            const p = parts[i].trim();
            if (p.toLowerCase().startsWith("lbw=")) {
                lbw = p.substring(4);
            } else if (!isNaN(parseFloat(p)) && weight === null) {
                weight = parseFloat(p);
            } else if (p.includes(",") || ["COMPOSITION", "FACE", "STYLE", "TEXTURE", "MIDD", "INS", "OUTS", "RESET"].includes(p.toUpperCase())) {
                lbw = p;
            } else if (i >= 2 && isNaN(parseFloat(p))) {
                lbw = p;
            }
        }

        // If no explicit weight was in tag (e.g. <lora:name>), weight defaults to 1.0
        if (weight === null) {
            weight = 1.0;
        }

        return {
            fullTag: tagStr,
            startIdx: openIdx,
            endIdx: closeIdx + 1,
            name: name,
            weight: weight,
            lbw: lbw
        };
    }

    /**
     * Handle click or key events on prompt textareas.
     */
    function handleCaretCheck(textarea) {
        if (!textarea || typeof textarea.value !== "string") return;
        const match = findLoraAtCursor(textarea);
        if (match) {
            showPopover(match, textarea);
        } else {
            hidePopover();
        }
    }

    /**
     * Check if an element is a prompt textarea or contained within prompt containers.
     */
    function isPromptTextarea(elem) {
        if (!elem || elem.tagName !== "TEXTAREA") return false;
        if (elem.closest("#txt2img_prompt, #img2img_prompt, #txt2img_neg_prompt, #img2img_neg_prompt, .prompt")) return true;
        // Fallback: any textarea containing <lora:
        if (typeof elem.value === "string" && elem.value.includes("<lora:")) return true;
        return false;
    }

    /**
     * Initialize listeners on txt2img and img2img prompt textareas.
     */
    function initListeners() {
        loadPresetsFromServer();

        // 1. Direct binding for currently existing textareas
        const bindTextarea = (ta) => {
            if (!ta || ta.dataset.lbwPopoverBound) return;
            ta.dataset.lbwPopoverBound = "true";
            ta.addEventListener("click", () => handleCaretCheck(ta));
            ta.addEventListener("keyup", (e) => {
                if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) {
                    handleCaretCheck(ta);
                }
            });
        };

        const root = typeof gradioApp === "function" ? gradioApp() : document;
        const selector = "#txt2img_prompt textarea, #img2img_prompt textarea, #txt2img_neg_prompt textarea, #img2img_neg_prompt textarea, .prompt textarea";
        root.querySelectorAll(selector).forEach(bindTextarea);

        // 2. Event delegation on document: guarantees catching clicks even if DOM re-rendered or inside tabs
        document.addEventListener("click", (e) => {
            if (isPromptTextarea(e.target)) {
                handleCaretCheck(e.target);
            }
        });

        document.addEventListener("keyup", (e) => {
            if (isPromptTextarea(e.target) && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) {
                handleCaretCheck(e.target);
            }
        });

        // 3. Dismiss popover on outside click (excluding popover itself and active textarea)
        document.addEventListener("mousedown", (e) => {
            if (!popoverElem || popoverElem.style.display === "none") return;
            if (popoverElem.contains(e.target)) return;
            if (e.target && e.target.tagName === "TEXTAREA") return;
            hidePopover();
        });

        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && popoverElem && popoverElem.style.display !== "none") {
                hidePopover();
            }
        });
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(initListeners);
    } else {
        document.addEventListener("DOMContentLoaded", initListeners);
    }
    // Also run immediately if DOM is already ready
    if (document.readyState === "complete" || document.readyState === "interactive") {
        initListeners();
    }
})();
