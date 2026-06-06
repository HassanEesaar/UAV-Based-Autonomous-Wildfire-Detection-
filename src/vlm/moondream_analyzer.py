"""
Moondream2 VLM: generates natural-language scene descriptions
from wildfire incident images.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
from .prompts import SCENE_DESCRIPTION_PROMPT


class MoondreamAnalyzer:
    """
    Uses Moondream2 to describe wildfire scenes detected by YOLO.

    Args:
        model_id (str): HuggingFace model ID or local path.
        device (str): 'cpu', 'cuda', or 'mps'.
        revision (str): Model revision/branch.
    """

    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        device: str = "cpu",
        revision: str = "2025-01-09",
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
        ).to(device)
        self.model.eval()

    def describe(self, image_path: str, prompt: str = None) -> str:
        """
        Generate a natural-language description of a wildfire incident image.

        Args:
            image_path (str): Path to the image file.
            prompt (str): Optional custom prompt. Defaults to SCENE_DESCRIPTION_PROMPT.

        Returns:
            str: Model-generated scene description.
        """
        image = Image.open(image_path).convert("RGB")
        enc_image = self.model.encode_image(image)

        question = prompt or SCENE_DESCRIPTION_PROMPT
        answer = self.model.answer_question(enc_image, question, self.tokenizer)
        return answer

    def batch_describe(self, image_paths: list[str]) -> list[str]:
        """
        Describe multiple images and return a list of descriptions.

        Args:
            image_paths: List of file paths.

        Returns:
            list[str]: One description per image.
        """
        return [self.describe(p) for p in image_paths]
