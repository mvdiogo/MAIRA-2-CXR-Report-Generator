# test_main.py
import pytest
import requests
from PIL import Image
from unittest.mock import patch, MagicMock
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import os
import main  # Import the main script

# Constants for testing
FRONTAL_URL = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png"
LATERAL_URL = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-2001.png"
INDICATION = "Dyspnea."
COMPARISON = "None."
TECHNIQUE = "PA and lateral views of the chest."
FAKE_FINDINGS = "This is a fake radiology report."
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


@pytest.fixture
def mock_images():
    """Fixture to create mock PIL images."""
    mock_image = Image.new("RGB", (256, 256))  # Example image
    return mock_image

@pytest.fixture
def mock_model_and_processor():
    """Fixture to mock the model and processor."""

    # Mock the AutoModelForCausalLM and AutoProcessor
    mock_model = MagicMock(spec=AutoModelForCausalLM)
    mock_processor = MagicMock(spec=AutoProcessor)

    # Mock the model's generate method to return dummy output
    mock_output = torch.tensor([[0, 1, 2, 3, 4, 5]])  # Fake token IDs
    mock_model.generate.return_value = mock_output
    mock_model.device = torch.device("cpu")

    # Mock processor's methods
    mock_processor.format_and_preprocess_reporting_input.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    mock_processor.tokenizer.decode.return_value = FAKE_FINDINGS  # Mock decoding

    return mock_model, mock_processor


@patch("main.AutoModelForCausalLM.from_pretrained")
@patch("main.AutoProcessor.from_pretrained")
def test_model_loading(mock_auto_processor, mock_auto_model, mock_model_and_processor):
    """Test that the model and processor are loaded correctly."""
    mock_model, mock_processor = mock_model_and_processor

    mock_auto_model.return_value = mock_model
    mock_auto_processor.return_value = mock_processor

    # Set a dummy HF_TOKEN for testing
    os.environ["HF_TOKEN"] = "dummy_token"

    # Call the main script (part that loads the model)
    main.model = None
    main.processor = None
    # The easiest way to do this is to call if __name__ == "__main__":
    import main
    # Check that the model and processor were loaded
    assert main.model is not None
    assert main.processor is not None
    assert main.model == mock_model
    assert main.processor == mock_processor


@patch("main.download_image")
def test_download_image_success(mock_download_image):
    """Test successful image download."""
    mock_image = Image.new("RGB", (256, 256))  # Example image
    mock_download_image.return_value = mock_image
    image = main.download_image("http://example.com/image.png")
    assert isinstance(image, Image.Image)

@patch("main.requests.get")
def test_download_image_failure(mock_get):
    """Test image download failure."""
    mock_get.side_effect = requests.exceptions.RequestException("Download failed")
    with pytest.raises(gr.Error) as excinfo:
        main.download_image("http://example.com/image.png")
    assert "Failed to download image" in str(excinfo.value)


@patch("main.download_image")
@patch("main.AutoProcessor.from_pretrained")
@patch("main.AutoModelForCausalLM.from_pretrained")
def test_generate_findings(mock_auto_model, mock_auto_processor, mock_download_image, mock_model_and_processor, mock_images):
    """Test the generate_findings function."""
    mock_model, mock_processor = mock_model_and_processor
    mock_auto_model.return_value = mock_model
    mock_auto_processor.return_value = mock_processor

    mock_download_image.return_value = mock_images

    # Call generate_findings
    frontal_image, lateral_image, prediction = main.generate_findings(
        FRONTAL_URL, LATERAL_URL, INDICATION, COMPARISON, TECHNIQUE
    )

    # Assert that the functions were called and outputs are as expected
    mock_download_image.assert_called()
    mock_model.generate.assert_called()
    assert isinstance(frontal_image, Image.Image)
    assert isinstance(lateral_image, Image.Image)
    assert prediction == FAKE_FINDINGS


@patch("main.download_image")
@patch("main.AutoProcessor.from_pretrained")
@patch("main.AutoModelForCausalLM.from_pretrained")
def test_generate_findings_exception(mock_auto_model, mock_auto_processor, mock_download_image, mock_model_and_processor):
    """Test generate_findings function when an exception is raised."""
    mock_model, mock_processor = mock_model_and_processor
    mock_auto_model.return_value = mock_model
    mock_auto_processor.return_value = mock_processor
    mock_download_image.side_effect = Exception("Download failed")
    with pytest.raises(gr.Error) as excinfo:
        main.generate_findings(
            FRONTAL_URL, LATERAL_URL, INDICATION, COMPARISON, TECHNIQUE
        )
    assert "Error generating findings" in str(excinfo.value)