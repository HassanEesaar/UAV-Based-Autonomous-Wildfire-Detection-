"""
GPT-4o mini alert generator: fuses YOLO detections and VLM descriptions
into structured wildfire emergency alerts.
"""

from openai import OpenAI
from .prompt_templates import build_alert_prompt


class AlertGenerator:
    """
    Uses GPT-4o mini to generate structured emergency alerts from
    wildfire detection data and VLM scene descriptions.

    Args:
        api_key (str): OpenAI API key.
        model (str): OpenAI model name.
        max_tokens (int): Maximum response length.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 600,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        detections: list[dict],
        descriptions: list[str],
        location_info: dict = None,
    ) -> dict:
        """
        Generate a structured wildfire alert.

        Args:
            detections: List of YOLO detection dicts (class_name, confidence, bbox).
            descriptions: List of VLM scene description strings (up to 10).
            location_info: Optional dict with keys: lat, lon, place_name.

        Returns:
            dict with keys:
                - situation_summary (str)
                - alert_level (str): 'low' | 'medium' | 'high' | 'emergency'
                - emergency_guidance (str)
                - raw_response (str)
        """
        prompt = build_alert_prompt(detections, descriptions, location_info)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an emergency response AI for wildfire incidents. "
                        "Generate accurate, concise, and actionable alerts based on "
                        "drone detection data and visual analysis."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=0.3,
        )

        raw = response.choices[0].message.content
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> dict:
        """Parse structured fields from GPT response."""
        result = {
            "situation_summary": "",
            "alert_level": "high",
            "emergency_guidance": "",
            "raw_response": raw,
        }

        lines = raw.strip().split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if "Wildfire Situation Summary" in line:
                current_section = "situation_summary"
            elif "Emergency Response Guidance" in line:
                current_section = "emergency_guidance"
            elif "Alert Level" in line.lower():
                for level in ["emergency", "high", "medium", "low"]:
                    if level in line.lower():
                        result["alert_level"] = level
                        break
            elif current_section and line:
                result[current_section] += line + " "

        result["situation_summary"] = result["situation_summary"].strip()
        result["emergency_guidance"] = result["emergency_guidance"].strip()
        return result
