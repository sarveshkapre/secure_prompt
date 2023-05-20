# Secure Prompt

Secure Prompt is a library designed to secure Language Learning Models (LLMs) from prompt injection attacks. It uses a pluggable architecture, allowing for the easy addition of new checks as plugins.

## How to Install

To install Secure Prompt, run the following command:

```bash
pip install .
```

## How to Use

Here's a basic usage example:

```python
from secure_prompt import Defense

# Initialize the defense system
defense = Defense()

# Get user input and model output
input = '...'
output = '...'

# Run the defense checks
defense.defend(input, output)
```

Please note that the `input` and `output` in this example should be replaced with your actual user input and model output.

<br />

## Plugins

Secure Prompt currently includes the following plugins:

1. **Semantic Similarity Check**: This check compares the semantic similarity of the model's output with a predefined set of typical responses.

2. **Input Sanitization**: This check sanitizes user input, removing or escaping potentially harmful characters or strings.

3. **Heuristics-Based Filtering**: This check uses a set of heuristic rules or patterns indicative of an injection attack to filter user input.

Each plugin is a separate Python class that you can extend and modify to suit your needs.

<br />

## Extending Secure Prompt

To add a new check to Secure Prompt, create a new Python class in the `plugins` directory. This class should contain a method for performing the check. Then, add an instance of this class to the `checks` list in `defense.py`.
