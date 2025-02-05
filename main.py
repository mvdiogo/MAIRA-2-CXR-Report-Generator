try:
    from huggingface_hub import HfHubHTTPError
except ImportError:
    try:
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        HfHubHTTPError = Exception

from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import torch
import requests
from PIL import Image
import gradio as gr
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


def download_image(url: str) -> Image.Image:
    """
    Download the image from the given URL and return as a PIL Image.
    """
    try:
        response = requests.get(url, headers={"User-Agent": "MAIRA-2"}, stream=True)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")
    except Exception as e:
        raise gr.Error(f"Failed to download image from {url}: {str(e)}")


def generate_findings(
    frontal_url: str,
    lateral_url: str,
    indication: str,
    comparison: str,
    technique: str
):
    """
    1. Download the frontal & lateral images from the provided URLs.
    2. Prepare the inputs for the model using the keys expected by MAIRA‑2.
    3. Generate the findings from the model.
    4. Return the two images and the generated findings text.
    """
    try:
        # 1. Download images
        frontal_image = download_image(frontal_url)
        lateral_image = download_image(lateral_url)

        # 2. Prepare the inputs using format_and_preprocess_reporting_input
        processed_inputs = processor.format_and_preprocess_reporting_input(
            current_frontal=frontal_image,
            current_lateral=lateral_image,
            indication=indication,
            technique=technique,
            comparison=comparison,
            prior_frontal=None,
            prior_report=None,
            return_tensors="pt",
        )

        # Move all returned tensors to the model's device.
        processed_inputs = {k: v.to(model.device) for k, v in processed_inputs.items()}

        # 3. Generate the findings using the model.
        with torch.no_grad():
            output_decoding = model.generate(
                **processed_inputs,
                max_new_tokens=512,
                num_beams=3,
                early_stopping=True,
                use_cache=True,
            )

        # Assume the processor returns an "input_ids" key for the prompt tokens.
        prompt_length = processed_inputs["input_ids"].shape[-1]
        decoded_text = processor.tokenizer.decode(
            output_decoding[0][prompt_length:], skip_special_tokens=True
        ).lstrip()

        # Optionally, if the processor has a method to convert the decoded text into plain text, use it.
        prediction = decoded_text

        return frontal_image, lateral_image, prediction

    except Exception as e:
        raise gr.Error(f"Error generating findings: {str(e)}")


if __name__ == "__main__":
    # Check for Hugging Face token
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print(
            "Error: Hugging Face token not found.  Please set the HF_TOKEN environment variable."
        )
        print(
            "  1. Request access to the microsoft/maira-2 model on Hugging Face: https://huggingface.co/microsoft/maira-2"
        )
        print("  2. Get your Hugging Face token from: https://huggingface.co/settings/tokens")
        print("  3. Set the HF_TOKEN environment variable:")
        print("     - In your terminal: `export HF_TOKEN='your_token_here'` (Linux/macOS)")
        print("     - In Windows: `set HF_TOKEN=your_token_here`")
        exit()

    # Load model and processor
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/maira-2",
            trust_remote_code=True,
            token=hf_token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )
        processor = AutoProcessor.from_pretrained(
            "microsoft/maira-2", trust_remote_code=True, token=hf_token
        )
    except HfHubHTTPError as e:
        print(f"Error loading the model from Hugging Face Hub: {e}")
        print(
            "Please ensure you have requested and been granted access to the 'microsoft/maira-2' model."
        )
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.is_available():
        model.half()  # Use half-precision for faster inference

    app_name = "MAIRA-2 CXR Report Generator"
    app_description = """
    Enter URLs for the frontal and lateral chest X-ray images and relevant metadata.
    Click "Generate Findings" to see the automatic radiology report findings.
    """

    with gr.Blocks(title=app_name) as demo:
        gr.Markdown(f"## {app_name}")
        gr.Markdown(app_description)

        with gr.Row():
            frontal_url = gr.Textbox(
                label="Frontal Image URL",
                value="https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png",
            )
            lateral_url = gr.Textbox(
                label="Lateral Image URL",
                value="https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-2001.png",
            )

        indication = gr.Textbox(label="Indication", value="Dyspnea.")
        comparison = gr.Textbox(label="Comparison", value="None.")
        technique = gr.Textbox(label="Technique", value="PA and lateral views of the chest.")

        generate_button = gr.Button("Generate Findings")

        with gr.Row():
            frontal_image_out = gr.Image(label="Frontal Image", type="pil")
            lateral_image_out = gr.Image(label="Lateral Image", type="pil")
        result_text_out = gr.Textbox(label="Generated Findings", lines=6)

        generate_button.click(
            fn=generate_findings,
            inputs=[frontal_url, lateral_url, indication, comparison, technique],
            outputs=[frontal_image_out, lateral_image_out, result_text_out],
            concurrency_limit=1,
        )

    demo.launch(share=True)