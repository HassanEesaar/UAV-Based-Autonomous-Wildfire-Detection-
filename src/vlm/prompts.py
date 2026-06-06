"""
Prompt templates for the Moondream2 VLM scene analysis stage.
"""

SCENE_DESCRIPTION_PROMPT = (
    "Describe this wildfire image in detail. Include: "
    "(1) the type and severity of fire or smoke visible, "
    "(2) the location and spread direction, "
    "(3) any nearby structures, vegetation, or water bodies, "
    "(4) environmental conditions such as wind or dryness, "
    "(5) any visible threats to human life or property."
)

SPREAD_ASSESSMENT_PROMPT = (
    "Based on this image, assess the fire spread risk. "
    "Describe the direction the fire appears to be spreading, "
    "the density and color of smoke, and any terrain features "
    "that might accelerate or slow the fire."
)

INFRASTRUCTURE_PROMPT = (
    "Identify any structures, roads, power lines, or human settlements "
    "visible in or near this wildfire scene. "
    "Estimate approximate distance between the fire front and any structures."
)
