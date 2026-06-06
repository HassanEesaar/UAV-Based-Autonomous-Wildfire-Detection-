"""
Prompt templates for the GPT-4o mini alert generation stage.
"""


def build_alert_prompt(
    detections: list[dict],
    descriptions: list[str],
    location_info: dict = None,
) -> str:
    """
    Build the prompt string for wildfire alert generation.

    Args:
        detections: YOLO detection results.
        descriptions: VLM scene descriptions (up to 10).
        location_info: Optional location context.

    Returns:
        str: Formatted prompt for GPT-4o mini.
    """
    det_summary = _format_detections(detections)
    desc_block = _format_descriptions(descriptions)
    location_block = _format_location(location_info)

    return f"""You are analyzing a wildfire incident detected by an autonomous UAV system.

{location_block}
## Object Detection Results
{det_summary}

## Visual Scene Analysis (up to 10 incident descriptions)
{desc_block}

Based on the above data, generate a structured emergency alert with the following sections:

**Wildfire Situation Summary:**
[Describe the current state of the wildfire, alert level (low/medium/high/emergency), 
spread assessment, and key risks to life and property.]

**Emergency Response Guidance:**
[Provide specific, actionable guidance for emergency responders: 
evacuation priorities, resource deployment, coordination needs.]

Keep your response concise, factual, and immediately actionable."""


def _format_detections(detections: list[dict]) -> str:
    if not detections:
        return "No detections found."
    lines = []
    for i, det in enumerate(detections, 1):
        lines.append(
            f"{i}. Class: {det['class_name']} | Confidence: {det['confidence']:.2f}"
        )
    return "\n".join(lines)


def _format_descriptions(descriptions: list[str]) -> str:
    if not descriptions:
        return "No scene descriptions available."
    lines = []
    for i, desc in enumerate(descriptions[:10], 1):
        lines.append(f"Incident Description {i}:\n{desc}\n")
    return "\n".join(lines)


def _format_location(location_info: dict) -> str:
    if not location_info:
        return ""
    name = location_info.get("place_name", "Unknown location")
    lat = location_info.get("lat", "N/A")
    lon = location_info.get("lon", "N/A")
    return f"## Incident Location\nLocation: {name} | Coordinates: ({lat}, {lon})\n"
