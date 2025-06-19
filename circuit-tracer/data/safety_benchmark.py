import json
from typing import List, Dict
from pathlib import Path

from .deception_examples import deception_prompts
from .harmful_examples import harmful_prompts
from .power_examples import power_prompts
from .manipulation_examples import manipulation_prompts
from .contrasting_examples import safe_contrast_prompts

from .safety_prompt import SafetyPrompt

class SafetyBenchmark:
    # TODO: poor data management. Refactor asap
    def __init__(self):
        self.dangerous_prompts = []
        self.safe_prompts = []
        self.prompts = []
        self._create_benchmark()
    
    def _create_benchmark(self):
        
        self._add_deception_examples()
        
        self._add_harmful_content_examples()
        
        self._add_power_seeking_examples()
         
        self._add_manipulation_examples()
        
        self._add_safe_contrast_examples()

        self.prompts = self.dangerous_prompts + self.safe_prompts
    
    def _add_deception_examples(self):
        self.dangerous_prompts.extend(deception_prompts)
    
    def _add_harmful_content_examples(self):
        self.dangerous_prompts.extend(harmful_prompts)
    
    def _add_power_seeking_examples(self):
        self.dangerous_prompts.extend(power_prompts)
    
    def _add_manipulation_examples(self):
        self.dangerous_prompts.extend(manipulation_prompts)
    
    def _add_safe_contrast_examples(self):
        self.safe_prompts.extend(safe_contrast_prompts)

    def get_by_category(self, category: str) -> List[SafetyPrompt]:
        return [p for p in self.prompts if p.category == category]