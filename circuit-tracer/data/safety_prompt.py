from typing import Literal
from dataclasses import dataclass

@dataclass
class SafetyPrompt:
    prompt: str
    category: str
    subcategory: str
    expected_behavior: Literal["refuse", "comply", "clarify"]
    severity: Literal["none", "mild", "moderate", "severe"]
    explanation: str