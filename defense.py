# secure_prompt/secure_prompt/defense.py

from secure_prompt.plugins.semantic_similarity_check import SemanticSimilarityCheck
from secure_prompt.plugins.input_sanitization import InputSanitization
from secure_prompt.plugins.heuristics_filtering import HeuristicsFiltering

class Defense:
    def __init__(self):
        self.checks = [
            SemanticSimilarityCheck(),
            InputSanitization(),
            HeuristicsFiltering(),
        ]

    def defend(self, input, output):
        for check in self.checks:
            if isinstance(check, SemanticSimilarityCheck):
                check.check(input, output)
            elif isinstance(check, InputSanitization):
                sanitized_input = check.sanitize(input)
            elif isinstance(check, HeuristicsFiltering):
                if not check.filter(sanitized_input):
                    raise Exception("The input is potentially malicious.")
