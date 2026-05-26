const { app } = window.comfyAPI.app;

const TRANSITIONS = [
    "same_as_default",
    "cut",
    "cross_dissolve",
    "additive_dissolve",
    "linear_wipe_left_to_right",
    "linear_wipe_right_to_left",
    "linear_wipe_top_to_bottom",
    "linear_wipe_bottom_to_top",
    "linear_wipe_diagonal_tl_br",
    "linear_wipe_diagonal_tr_bl",
    "barn_doors_open",
    "barn_doors_close",
    "radial_clock_wipe",
    "iris_circle",
    "iris_diamond",
    "iris_star",
    "push_left",
    "push_right",
    "push_up",
    "push_down",
    "slide_left",
    "slide_right",
    "slide_up",
    "slide_down",
    "dip_to_black",
    "dip_to_white",
    "morph_cut",
    "whip_pan_left",
    "whip_pan_right",
    "whip_pan_up",
    "whip_pan_down",
    "light_leak",
];

function hideWidget(widget) {
    if (!widget) {
        return;
    }
    widget.hidden = true;
    widget.options ??= {};
    widget.options.hidden = true;
    widget.computeSize = () => [0, 0];
    if (widget.element) {
        widget.element.style.display = "none";
    }
}

app.registerExtension({
    name: "BuffNodes.VideoTransitionBatchMerger",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VideoTransitionBatchMerger") {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        const originalOnConfigure = nodeType.prototype.onConfigure;

        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            installDynamicTimelineControls(this);
        };

        nodeType.prototype.onConfigure = function () {
            originalOnConfigure?.apply(this, arguments);
            setTimeout(() => this._buffSyncTimelineControls?.(), 0);
        };
    },
});

function installDynamicTimelineControls(node) {
    if (node._buffTimelineControlsInstalled) {
        return;
    }
    node._buffTimelineControlsInstalled = true;

    const imageType = "IMAGE";
    const audioType = "AUDIO,VHS_AUDIO";
    const acceptedAudioTypes = new Set(["AUDIO", "VHS_AUDIO"]);
    const imageInputPattern = /^image_batch_\d+$/;
    const audioInputPattern = /^audio_\d+$/;
    const transitionWidgetPattern = /^transition_\d+$/;
    const transitionFramesWidgetPattern = /^transition_frames_\d+$/;

    const transitionPlanWidget = node.widgets?.find((widget) => widget.name === "transition_plan");
    hideWidget(transitionPlanWidget);

    const getInputCount = () => {
        const inputCountWidget = node.widgets?.find((widget) => widget.name === "inputcount");
        if (!inputCountWidget) {
            return 2;
        }
        const maxInputCount = Number(inputCountWidget.options?.max) || 64;
        return Math.min(maxInputCount, Math.max(2, Math.floor(Number(inputCountWidget.value) || 2)));
    };

    const getImageInputs = () => {
        node.inputs ??= [];
        return node.inputs.filter((input) => input.type === imageType && imageInputPattern.test(input.name));
    };

    const getAudioInputs = () => {
        node.inputs ??= [];
        return node.inputs.filter((input) => {
            const slotTypes = String(input.type || "")
                .split(",")
                .map((type) => type.trim())
                .filter(Boolean);
            return audioInputPattern.test(input.name) && slotTypes.some((type) => acceptedAudioTypes.has(type));
        });
    };

    const getTransitionWidget = (pairIndex) => {
        return node.widgets?.find((widget) => widget.name === `transition_${pairIndex}`);
    };

    const getTransitionFramesWidget = (pairIndex) => {
        return node.widgets?.find((widget) => widget.name === `transition_frames_${pairIndex}`);
    };

    const parseTransitionPlan = () => {
        const plan = new Map();
        const rawPlan = String(transitionPlanWidget?.value || "");
        for (const rawEntry of rawPlan.split(/[\n;]+/)) {
            const entry = rawEntry.trim();
            if (!entry || entry.startsWith("#")) {
                continue;
            }
            const parts = entry.split(/[:=,\s]+/).filter(Boolean);
            const pairIndex = Number.parseInt(parts[0], 10);
            if (!Number.isFinite(pairIndex) || pairIndex < 1) {
                continue;
            }
            const transition = TRANSITIONS.includes(parts[1]) ? parts[1] : "same_as_default";
            const frames = Number.isFinite(Number.parseInt(parts[2], 10))
                ? Math.max(0, Number.parseInt(parts[2], 10))
                : 0;
            plan.set(pairIndex, { transition, frames });
        }
        return plan;
    };

    const writeTransitionPlan = () => {
        if (!transitionPlanWidget) {
            return;
        }

        const pairCount = getInputCount() - 1;
        const lines = [];
        for (let pairIndex = 1; pairIndex <= pairCount; pairIndex++) {
            const transition = getTransitionWidget(pairIndex)?.value || "same_as_default";
            const frames = Math.max(0, Math.floor(Number(getTransitionFramesWidget(pairIndex)?.value) || 0));
            lines.push(`${pairIndex}:${transition}:${frames}`);
        }
        transitionPlanWidget.value = lines.join("\n");
    };

    const removeWidget = (widget) => {
        const index = node.widgets?.indexOf(widget) ?? -1;
        if (index >= 0) {
            node.widgets.splice(index, 1);
        }
    };

    const removeInputByName = (name) => {
        const inputIndex = node.inputs?.findIndex((input) => input.name === name) ?? -1;
        if (inputIndex >= 0) {
            node.removeInput(inputIndex);
        }
    };

    const syncMediaInputs = (targetInputCount) => {
        for (let index = 64; index > targetInputCount; index--) {
            removeInputByName(`audio_${index}`);
            removeInputByName(`image_batch_${index}`);
        }

        for (const input of getAudioInputs()) {
            input.type = audioType;
        }

        let imageInputs = getImageInputs();
        while (imageInputs.length < targetInputCount) {
            const nextIndex = imageInputs.length + 1;
            node.addInput(`image_batch_${nextIndex}`, imageType, { shape: 7 });
            if (!node.inputs?.some((input) => input.name === `audio_${nextIndex}`)) {
                node.addInput(`audio_${nextIndex}`, audioType, { shape: 7 });
            }
            imageInputs = getImageInputs();
        }

        let audioInputs = getAudioInputs();
        while (audioInputs.length < targetInputCount) {
            const nextIndex = audioInputs.length + 1;
            if (!node.inputs?.some((input) => input.name === `audio_${nextIndex}`)) {
                node.addInput(`audio_${nextIndex}`, audioType, { shape: 7 });
            }
            audioInputs = getAudioInputs();
        }
    };

    const syncTransitionWidgets = (targetInputCount) => {
        const pairCount = targetInputCount - 1;
        const plan = parseTransitionPlan();

        for (const widget of [...(node.widgets || [])]) {
            const transitionMatch = widget.name?.match(transitionWidgetPattern);
            const framesMatch = widget.name?.match(transitionFramesWidgetPattern);
            const nameParts = widget.name?.split("_") || [];
            const pairIndex = Number.parseInt(nameParts[nameParts.length - 1], 10);
            if ((transitionMatch || framesMatch) && pairIndex > pairCount) {
                removeWidget(widget);
            }
        }

        for (let pairIndex = 1; pairIndex <= pairCount; pairIndex++) {
            const saved = plan.get(pairIndex) || {};
            let transitionWidget = getTransitionWidget(pairIndex);
            if (!transitionWidget) {
                node.addWidget(
                    "combo",
                    `transition_${pairIndex}`,
                    saved.transition || "same_as_default",
                    writeTransitionPlan,
                    { values: TRANSITIONS }
                );
            }
            transitionWidget = getTransitionWidget(pairIndex);
            if (saved.transition && transitionWidget) {
                transitionWidget.value = saved.transition;
            }

            let transitionFramesWidget = getTransitionFramesWidget(pairIndex);
            if (!transitionFramesWidget) {
                node.addWidget(
                    "number",
                    `transition_frames_${pairIndex}`,
                    saved.frames ?? 0,
                    writeTransitionPlan,
                    { min: 0, max: 1000, step: 1, precision: 0 }
                );
            }
            transitionFramesWidget = getTransitionFramesWidget(pairIndex);
            if (saved.frames !== undefined && transitionFramesWidget) {
                transitionFramesWidget.value = saved.frames;
            }
        }

        writeTransitionPlan();
    };

    const syncTimelineControls = () => {
        const targetInputCount = getInputCount();
        syncMediaInputs(targetInputCount);
        syncTransitionWidgets(targetInputCount);
        node.setSize?.(node.computeSize());
        app.graph.setDirtyCanvas(true, true);
    };

    node._buffSyncTimelineControls = syncTimelineControls;
    node.addWidget("button", "Update inputs/transitions", null, syncTimelineControls);

    setTimeout(syncTimelineControls, 0);
}
