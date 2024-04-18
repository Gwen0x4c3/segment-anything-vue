const canvasScaleInitializer = ({
                                    width,
                                    height,
                                    containerRef,
                                    shouldFitToWidth,
                                }) => {
    const containerWidth = containerRef.offsetWidth || width;
    const containerHeight = containerRef.offsetHeight || height;
    return canvasScaleResizer({
        width,
        height,
        containerWidth,
        containerHeight,
        shouldFitToWidth,
    });
};

const canvasScaleResizer = ({
                                width,
                                height,
                                containerWidth,
                                containerHeight,
                                shouldFitToWidth,
                            }) => {
    let scale = 1;
    const xScale = containerWidth / width;
    const yScale = containerHeight / height;
    if (shouldFitToWidth) {
        scale = xScale;
    } else {
        scale = Math.min(xScale, yScale);
    }
    const scaledWidth = scale * width;
    const scaledHeight = scale * height;
    const scalingStyle = {
        transform: `scale(${scale})`,
        transformOrigin: "left top",
    };
    const scaledDimensionsStyle = {
        width: scaledWidth,
        height: scaledHeight,
    };
    return {
        scalingStyle,
        scaledDimensionsStyle,
        scaledWidth,
        scaledHeight,
        containerWidth,
        containerHeight,
    };
};

export { canvasScaleInitializer, canvasScaleResizer };
