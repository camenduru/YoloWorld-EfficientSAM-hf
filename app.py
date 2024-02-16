from typing import List

import gradio as gr
import numpy as np
import supervision as sv
from inference.models import YOLOWorld

MARKDOWN = """
# YOLO-World 🌎

Powered by Roboflow [Inference](https://github.com/roboflow/inference) and [Supervision](https://github.com/roboflow/supervision).
"""

MODEL = YOLOWorld(model_id="yolo_world/l")
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)


def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(',')]


def process_image(input_image: np.ndarray, categories: str) -> np.ndarray:
    categories = process_categories(categories)
    MODEL.set_classes(categories)
    results = MODEL.infer(input_image, confidence=0.003)
    detections = sv.Detections.from_inference(results).with_nms(0.1)
    output_image = input_image.copy()
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
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
            scale=5
        )
        submit_button_component = gr.Button('Submit', scale=1)

    submit_button_component.click(
        fn=process_image,
        inputs=[input_image_component, categories_text_component],
        outputs=output_image_component
    )

demo.launch(debug=False, show_error=True)
