from typing import List

import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from inference.models import YOLOWorld

from utils.efficient_sam import load, inference_with_box

MARKDOWN = """
# YOLO-World + EfficientSAM 🔥

This is a demo of zero-shot instance segmentation using 
[YOLO-World](https://github.com/AILab-CVC/YOLO-World) and 
[EfficientSAM](https://github.com/yformer/EfficientSAM).

Powered by Roboflow [Inference](https://github.com/roboflow/inference) and 
[Supervision](https://github.com/roboflow/supervision).
"""

EXAMPLES = [
    ['https://media.roboflow.com/dog.jpeg', 'dog, eye, nose, tongue, car', 0.005, 0.1, True, False, False],
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EFFICIENT_SAM_MODEL = load(device=DEVICE)
YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/l")

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(',')]


def process_image(
    input_image: np.ndarray,
    categories: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    with_segmentation: bool = True,
    with_confidence: bool = False,
    with_class_agnostic_nms: bool = False,
) -> np.ndarray:
    categories = process_categories(categories)
    YOLO_WORLD_MODEL.set_classes(categories)
    results = YOLO_WORLD_MODEL.infer(input_image, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results)
    detections = detections.with_nms(
        class_agnostic=with_class_agnostic_nms,
        threshold=iou_threshold)
    if with_segmentation:
        masks = []
        for [x_min, y_min, x_max, y_max] in detections.xyxy:
            box = np.array([[x_min, y_min], [x_max, y_max]])
            mask = inference_with_box(input_image, box, EFFICIENT_SAM_MODEL, DEVICE)
            masks.append(mask)
        detections.mask = np.array(masks)

    labels = [
        (
            f"{categories[class_id]}: {confidence:.2f}"
            if with_confidence
            else f"{categories[class_id]}"
        )
        for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]
    output_image = input_image.copy()
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return output_image


confidence_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.3,
    step=0.01,
    label="Confidence Threshold",
    info=(
        "The confidence threshold for the YOLO-World model. Lower the threshold to "
        "reduce false negatives, enhancing the model's sensitivity to detect "
        "sought-after objects. Conversely, increase the threshold to minimize false "
        "positives, preventing the model from identifying objects it shouldn't."
    ))

iou_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.5,
    step=0.01,
    label="IoU Threshold",
    info=(
        "The Intersection over Union (IoU) threshold for non-maximum suppression. "
        "Decrease the value to lessen the occurrence of overlapping bounding boxes, "
        "making the detection process stricter. On the other hand, increase the value "
        "to allow more overlapping bounding boxes, accommodating a broader range of "
        "detections."
    ))

with_segmentation_component = gr.Checkbox(
    value=True,
    label="With Segmentation",
    info=(
        "Whether to run EfficientSAM for instance segmentation."
    )
)

with_confidence_component = gr.Checkbox(
    value=False,
    label="Display Confidence",
    info=(
        "Whether to display the confidence of the detected objects."
    )
)

with_class_agnostic_nms_component = gr.Checkbox(
    value=False,
    label="Use Class-Agnostic NMS",
    info=(
        "Suppress overlapping bounding boxes across all classes."
    )
)


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Accordion("Configuration", open=False):
        confidence_threshold_component.render()
        iou_threshold_component.render()
        with gr.Row():
            with_segmentation_component.render()
            with_confidence_component.render()
            with_class_agnostic_nms_component.render()
    with gr.Row():
        input_image_component = gr.Image(
            type='numpy',
            label='Input Image'
        )
        output_image_component = gr.Image(
            type='numpy',
            label='Output Image'
        )
    with gr.Row():
        categories_text_component = gr.Textbox(
            label='Categories',
            placeholder='comma separated list of categories',
            scale=7
        )
        submit_button_component = gr.Button(
            value='Submit',
            scale=1,
            variant='primary'
        )
    gr.Examples(
        fn=process_image,
        examples=EXAMPLES,
        inputs=[
            input_image_component,
            categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_segmentation_component,
            with_confidence_component,
            with_class_agnostic_nms_component
        ],
        outputs=output_image_component
    )

    submit_button_component.click(
        fn=process_image,
        inputs=[
            input_image_component,
            categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_segmentation_component,
            with_confidence_component,
            with_class_agnostic_nms_component
        ],
        outputs=output_image_component
    )

demo.launch(debug=False, show_error=True)
