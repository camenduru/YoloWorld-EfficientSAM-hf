from typing import List

import os
import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from inference.models import YOLOWorld

from utils.efficient_sam import load, inference_with_boxes
from utils.video import generate_file_name, calculate_end_frame_index, create_directory

MARKDOWN = """
# YOLO-World + EfficientSAM 🔥

This is a demo of zero-shot object detection and instance segmentation using 
[YOLO-World](https://github.com/AILab-CVC/YOLO-World) and 
[EfficientSAM](https://github.com/yformer/EfficientSAM).

Powered by Roboflow [Inference](https://github.com/roboflow/inference) and 
[Supervision](https://github.com/roboflow/supervision).
"""

RESULTS = "results"

IMAGE_EXAMPLES = [
    ['https://media.roboflow.com/dog.jpeg', 'dog, eye, nose, tongue, car', 0.005, 0.1, True, False, False],
]
VIDEO_EXAMPLES = [
    ['https://media.roboflow.com/supervision/video-examples/croissant-1280x720.mp4', 'croissant', 0.01, 0.2, False, False, False],
    ['https://media.roboflow.com/supervision/video-examples/suitcases-1280x720.mp4', 'suitcase', 0.1, 0.2, False, False, False],
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EFFICIENT_SAM_MODEL = load(device=DEVICE)
YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/l")

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


create_directory(directory_path=RESULTS)


def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(',')]


def annotate_image(
    input_image: np.ndarray,
    detections: sv.Detections,
    categories: List[str],
    with_confidence: bool = False,
) -> np.ndarray:
    labels = [
        (
            f"{categories[class_id]}: {confidence:.3f}"
            if with_confidence
            else f"{categories[class_id]}"
        )
        for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]
    output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image


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
        threshold=iou_threshold
    )
    if with_segmentation:
        detections.mask = inference_with_boxes(
            image=input_image,
            xyxy=detections.xyxy,
            model=EFFICIENT_SAM_MODEL,
            device=DEVICE
        )
    output_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    output_image = annotate_image(
        input_image=output_image,
        detections=detections,
        categories=categories,
        with_confidence=with_confidence
    )
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)


def process_video(
    input_video: str,
    categories: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    with_segmentation: bool = True,
    with_confidence: bool = False,
    with_class_agnostic_nms: bool = False,
    progress=gr.Progress(track_tqdm=True)
) -> str:
    categories = process_categories(categories)
    YOLO_WORLD_MODEL.set_classes(categories)
    video_info = sv.VideoInfo.from_video_path(input_video)
    total = calculate_end_frame_index(input_video)
    frame_generator = sv.get_video_frames_generator(
        source_path=input_video,
        end=total
    )
    result_file_name = generate_file_name(extension="mp4")
    result_file_path = os.path.join(RESULTS, result_file_name)
    with sv.VideoSink(result_file_path, video_info=video_info) as sink:
        for _ in tqdm(range(total), desc="Processing video..."):
            frame = next(frame_generator)
            results = YOLO_WORLD_MODEL.infer(frame, confidence=confidence_threshold)
            detections = sv.Detections.from_inference(results)
            detections = detections.with_nms(
                class_agnostic=with_class_agnostic_nms,
                threshold=iou_threshold
            )
            if with_segmentation:
                detections.mask = inference_with_boxes(
                    image=frame,
                    xyxy=detections.xyxy,
                    model=EFFICIENT_SAM_MODEL,
                    device=DEVICE
            )
            frame = annotate_image(
                input_image=frame,
                detections=detections,
                categories=categories,
                with_confidence=with_confidence
            )
            sink.write_frame(frame)
    return result_file_path


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
    with gr.Tab(label="Image"):
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
            image_categories_text_component = gr.Textbox(
                label='Categories',
                placeholder='comma separated list of categories',
                scale=7
            )
            image_submit_button_component = gr.Button(
                value='Submit',
                scale=1,
                variant='primary'
            )
        gr.Examples(
            fn=process_image,
            examples=IMAGE_EXAMPLES,
            inputs=[
                input_image_component,
                image_categories_text_component,
                confidence_threshold_component,
                iou_threshold_component,
                with_segmentation_component,
                with_confidence_component,
                with_class_agnostic_nms_component
            ],
            outputs=output_image_component
        )
    with gr.Tab(label="Video"):
        with gr.Row():
            input_video_component = gr.Video(
                label='Input Video'
            )
            output_video_component = gr.Video(
                label='Output Video'
            )
        with gr.Row():
            video_categories_text_component = gr.Textbox(
                label='Categories',
                placeholder='comma separated list of categories',
                scale=7
            )
            video_submit_button_component = gr.Button(
                value='Submit',
                scale=1,
                variant='primary'
            )
        gr.Examples(
            fn=process_video,
            examples=VIDEO_EXAMPLES,
            inputs=[
                input_video_component,
                video_categories_text_component,
                confidence_threshold_component,
                iou_threshold_component,
                with_segmentation_component,
                with_confidence_component,
                with_class_agnostic_nms_component
            ],
            outputs=output_image_component
        )

    image_submit_button_component.click(
        fn=process_image,
        inputs=[
            input_image_component,
            image_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_segmentation_component,
            with_confidence_component,
            with_class_agnostic_nms_component
        ],
        outputs=output_image_component
    )
    video_submit_button_component.click(
        fn=process_video,
        inputs=[
            input_video_component,
            video_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_segmentation_component,
            with_confidence_component,
            with_class_agnostic_nms_component
        ],
        outputs=output_video_component
    )

demo.launch(debug=True, show_error=True, share=True)
