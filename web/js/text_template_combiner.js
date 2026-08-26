const { app } = window.comfyAPI.app;

app.registerExtension({
    name: "BuffNodes.TextTemplateCombiner",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TextTemplateCombiner") {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        const originalOnConfigure = nodeType.prototype.onConfigure;

        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            installDynamicTextInputs(this);
        };

        nodeType.prototype.onConfigure = function () {
            originalOnConfigure?.apply(this, arguments);
            setTimeout(() => this._buffSyncTextTemplateInputs?.(), 0);
        };
    },
});

function installDynamicTextInputs(node) {
    if (node._buffTextTemplateInputsInstalled) {
        return;
    }
    node._buffTextTemplateInputsInstalled = true;

    const maxInputs = 20;
    const stringType = "STRING";
    const textInputPattern = /^text_\d+$/;
    const placeholderPattern = /(?<!\{)\{([1-9]\d*)\}(?!\})/g;
    const choicePattern = /(?<!\{)\{([^{}\n]*\|[^{}\n]*)\}(?!\})/g;

    const getWidget = (name) => node.widgets?.find((widget) => widget.name === name);

    const getManualInputCount = () => {
        const inputCountWidget = getWidget("inputcount");
        if (!inputCountWidget) {
            return 1;
        }
        const maxInputCount = Number(inputCountWidget.options?.max) || maxInputs;
        return Math.min(maxInputCount, Math.max(0, Math.floor(Number(inputCountWidget.value) || 0)));
    };

    const getPlaceholderInputCount = () => {
        const mainWidget = getWidget("main");
        const text = String(mainWidget?.value || "");
        let maxIndex = 0;
        for (const match of text.matchAll(placeholderPattern)) {
            const index = Number.parseInt(match[1], 10);
            if (Number.isFinite(index) && index <= maxInputs) {
                maxIndex = Math.max(maxIndex, index);
            }
        }
        for (const match of text.matchAll(choicePattern)) {
            for (const option of match[1].split("|")) {
                const index = Number.parseInt(option.trim(), 10);
                if (String(index) === option.trim() && index >= 1 && index <= maxInputs) {
                    maxIndex = Math.max(maxIndex, index);
                }
            }
        }
        return maxIndex;
    };

    const getTargetInputCount = () => {
        return Math.max(getManualInputCount(), getPlaceholderInputCount());
    };

    const getTextInputs = () => {
        node.inputs ??= [];
        return node.inputs.filter((input) => textInputPattern.test(input.name));
    };

    const removeInputByName = (name) => {
        const inputIndex = node.inputs?.findIndex((input) => input.name === name) ?? -1;
        if (inputIndex >= 0) {
            node.removeInput(inputIndex);
            return true;
        }
        return false;
    };

    const growToFitSockets = () => {
        const currentSize = node.size ? [...node.size] : [0, 0];
        const computedSize = node.computeSize?.();
        if (!computedSize) {
            return;
        }

        const nextSize = [
            Math.max(currentSize[0] || 0, computedSize[0] || 0),
            Math.max(currentSize[1] || 0, computedSize[1] || 0),
        ];

        if (nextSize[0] !== currentSize[0] || nextSize[1] !== currentSize[1]) {
            node.setSize?.(nextSize);
        }
    };

    const syncInputs = () => {
        const targetInputCount = getTargetInputCount();
        const inputCountWidget = getWidget("inputcount");
        if (inputCountWidget && Number(inputCountWidget.value) < targetInputCount) {
            inputCountWidget.value = targetInputCount;
        }

        let socketsChanged = false;
        for (let index = maxInputs; index > targetInputCount; index--) {
            socketsChanged = removeInputByName(`text_${index}`) || socketsChanged;
        }

        for (const input of getTextInputs()) {
            input.type = stringType;
        }

        for (let index = 1; index <= targetInputCount; index++) {
            if (!node.inputs?.some((input) => input.name === `text_${index}`)) {
                node.addInput(`text_${index}`, stringType, { shape: 7 });
                socketsChanged = true;
            }
        }

        if (socketsChanged) {
            growToFitSockets();
        }
        app.graph.setDirtyCanvas(true, true);
    };

    const wrapWidgetCallback = (widget) => {
        if (!widget || widget._buffTextTemplateCallbackWrapped) {
            return;
        }
        widget._buffTextTemplateCallbackWrapped = true;
        const originalCallback = widget.callback;
        widget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            setTimeout(syncInputs, 0);
            return result;
        };
    };

    node._buffSyncTextTemplateInputs = syncInputs;
    node.addWidget("button", "Update text inputs", null, syncInputs);
    wrapWidgetCallback(getWidget("main"));
    wrapWidgetCallback(getWidget("inputcount"));

    setTimeout(syncInputs, 0);
}
