import re
from html import escape

class InputSanitization:
    def __init__(self):
        self.harmful_patterns = [
            r"[0-9a-f]{32}",
            r"\b(password|token|secret)\b",
        ]
        self.harmful_symbols = ["{", "}", "<", ">", "$"]
        self.compiled_patterns = [re.compile(pattern) for pattern in self.harmful_patterns]

    def sanitize(self, input):
        sanitized_input = escape(input)
        for symbol in self.harmful_symbols:
            sanitized_input = sanitized_input.replace(symbol, "")
        return sanitized_input

    def validate(self, input):
        for compiled_pattern in self.compiled_patterns:
            if compiled_pattern.search(input):
                return False
        return True

    def process_prompt(self, input):
        sanitized_input = self.sanitize(input)
        if not self.validate(sanitized_input):
            raise Exception("The input contains harmful content.")
        return sanitized_input
