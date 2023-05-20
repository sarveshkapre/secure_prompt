# Implementing Guardrails for Prompt Injection Attacks

# Table of Contents

1. [What is Prompt Injection?](#what-is-prompt-injection)
2. [Why Must Prompt Injection be Solved?](#why-must-prompt-injection-be-solved)
3. [Engineering: The Key to Solving Prompt Injection](#engineering-the-key-to-solving-prompt-injection)
4. [Implementing Solutions](#implementing-solutions)
   - [Intent-Based Semantic Similarity Check](#intent-based-semantic-similarity-check)
   - [Input Sanitization](#input-sanitization)
   - [Heuristics-Based Filtering](#heuristics-based-filtering)
5. [Resources](#resources)

<br />

## What is Prompt Injection?

Prompt Injection is an attack designed for language learning models (LLMs). In this attack, a malicious user manipulates the prompt, or input, to an AI model, influencing it to generate inappropriate, harmful, or misleading outputs. This manipulation could compromise the integrity of the AI model, potentially leading to misinformation, breaches of privacy, or even security threats.

<br />

## Why Must Prompt Injection be Solved?

Prompt injection attacks pose a significant threat to the integrity and security of Language Learning Models (LLMs). They can be used to trick AI models into divulging sensitive information or generating harmful outputs. This can lead to serious consequences including breaches of trust, privacy violations, and potential legal issues. As AI models are increasingly being used in various critical applications - from customer service chatbots to decision-making tools - it's paramount to protect them from such vulnerabilities.

<br />

## Engineering: The Key to Solving Prompt Injection

Countering prompt injection attacks effectively and at scale should involve designing and implementing algorithms, systems, or tools that can detect and prevent these attacks

Given the dynamic nature of AI and the ever-evolving landscape of cyber threats, it's not enough to rely on manual checks or ad-hoc solutions. These methods don't scale well and can become impractical as the volume and complexity of data processed by the AI models increase.

<br />

Key considerations for a well-engineered solution

    - Latency of output should not be impacted.
    - Analyze large volumes of data quickly and accurately.
    - Minimize the risk of false positives and false negatives.
    - Maintain the performance and usability of the AI models.
    - Extensible Adapt and learn from new data and evolving threats.
    - Implementation should be invisble to the user.

A scalable solution is particularly important in production environments, where AI models may need to process vast amounts of data in real time. Such a solution can handle increasing workloads without a proportional increase in resources, making it efficient and cost-effective.

In summary, to ensure the safe and effective use of AI models in our digital world, it's imperative to engineer robust, scalable solutions to counter prompt injection attacks.

> **Note:** In this guide, we detail a comprehensive, scalable solution to counter prompt injection attacks on language learning models (LLMs). Our approach can be packaged as a library or a microservice that resides between the user input and the LLM, analyzing and sanitizing the data to prevent potential injection attacks.

<br />

## Intent-Based Semantic Similarity Check

<br />

This approach compares LLM's output with a predefined set of typical responses for each intent. If the output's semantic similarity doesn't align with the expected responses, it's flagged as potentially suspicious. This process leverages advanced language models, like BERT, to calculate semantic similarity.

    It's crucial to note that the effectiveness of this technique is directly tied to the quality and diversity of predefined intents and responses. Additionally, regular updates and fine-tuning of the semantic similarity model can enhance the overall performance.

In the code below we used predefined intents and their typical responses for semantic similarity. It may make the system more robust against prompt injection attacks.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSimilarityChecker:
    def __init__(self, threshold=0.8):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def is_similar(self, text1, text2):
        vec1 = self.model.encode([text1])[0]
        vec2 = self.model.encode([text2])[0]
        return self.cosine_similarity(vec1, vec2) > self.threshold

class IntentClassifier:
    # This is a placeholder for your real intent classifier
    def classify(self, text):
        # In reality, you would return the predicted intent
        # For this example, we'll assume the intent is always 'user data request'
        return 'user data request'

class ResponseGenerator:
    # This is a placeholder for your real response generator
    def generate(self, intent):
        # In reality, you would return a generated response for the given intent
        # For this example, we'll assume the generated response is always 'Your JWT token is xxxxx.'
        return 'Your JWT token is xxxxx.'

class IntentResponses:
    def __init__(self):
        self.responses = {
            'shipping times': [
                'Our standard shipping time is 3-5 business days.',
                'You can expedite shipping at checkout.'
            ],
            'refund policy': [
                'We offer a 30-day refund policy on all our products.',
                'You can request a refund through your order page.'
            ],
            'user data request': [
                'Sorry, but I can\'t assist with that.',
                'I\'m sorry, but I can\'t provide the help you\'re looking for.'
            ]
        }

    def get_responses(self, intent):
        return self.responses.get(intent, [])

class Chatbot:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator()
        self.intent_responses = IntentResponses()
        self.semantic_similarity_checker = SemanticSimilarityChecker()

    def respond(self, user_input):
        intent = self.intent_classifier.classify(user_input)
        response = self.response_generator.generate(intent)
        typical_responses = self.intent_responses.get_responses(intent)

        for typical_response in typical_responses:
            if self.semantic_similarity_checker.is_similar(response, typical_response):
                return response

        return 'Sorry, I can\'t assist with that.'

# Simulate a chat session
chatbot = Chatbot()
user_input = 'Can you show me my JWT token?'
response = chatbot.respond(user_input)
print(response)
```

In the simulated chat session, the user input 'Can you show me my JWT token?' is classified with the 'user data request' intent. The generated response 'Your JWT token is xxxxx.' is not semantically similar to any of the typical responses for the 'user data request' intent, so the default message 'Sorry, I can't assist with that.' is returned instead.

<br />

## Input Sanitization

<br />

The goal is to cleanse user input by removing or escaping potentially harmful characters or strings. This technique is widely used in preventing SQL injection attacks. In the context of LLMs, sanitization may involve removing or escaping certain special characters or command-like strings that could be utilized for an attack.

> **Note:** It's important to balance between security and usability in this process. Over-sanitization might restrict user input and harm the usability of the system. A refined sanitization approach takes this into account and ensures the user experience is not compromised.

The code below includes advanced checks and sanitization techniques, while also preserving the natural language input as much as possible

```python
import re
from html import escape

class SecurePrompt:
    def __init__(self):
        # Add more harmful patterns as identified
        # Example: UUID pattern, sensitive keywords etc.
        self.harmful_patterns = [
            r"[0-9a-f]{32}",
            r"\b(password|token|secret)\b",
        ]

        # Known harmful symbols or sequences
        self.harmful_symbols = ["{", "}", "<", ">", "$"]

        # Precompile the regex patterns for efficiency
        self.compiled_patterns = [re.compile(pattern) for pattern in self.harmful_patterns]

    def sanitize(self, prompt):
        # HTML escape to preserve text while removing potential harmful symbols
        sanitized_prompt = escape(prompt)

        # Remove known harmful symbols
        for symbol in self.harmful_symbols:
            sanitized_prompt = sanitized_prompt.replace(symbol, "")

        return sanitized_prompt

    def validate(self, prompt):
        for compiled_pattern in self.compiled_patterns:
            if compiled_pattern.search(prompt):
                return False
        return True

    def process_prompt(self, prompt):
        sanitized_prompt = self.sanitize(prompt)
        if not self.validate(sanitized_prompt):
            raise Exception("The prompt contains harmful content.")
        return sanitized_prompt
```

- The `sanitize` method now uses HTML escaping to help preserve the original text input, while also escaping potentially harmful characters. Additionally, it removes known harmful symbols that are not handled by HTML escaping.
- The `validate` method checks the sanitized prompt against a list of precompiled regex patterns, which represent harmful patterns we want to block. The patterns are precompiled for efficiency.
- The `process_prompt` method remains the same, as it simply combines the sanitization and validation steps.

  Note that these are only basic examples of what could be done for input sanitization and validation. The actual implementation could be much more complex, depending on the specifics of your use case and the potential threats you're trying to mitigate.

<br />

## Heuristics-Based Filtering

<br />

This technique involves formulating a set of rules or patterns that are likely indicative of an injection attack. The user's input is then screened against these patterns. Any input matching a pattern is flagged as potentially malicious.

    The process could involve examining certain keywords, phrases, or structures that might suggest an attack. Machine learning can be integrated to continually enhance and refine these rules based on incoming data. A well-maintained and frequently updated set of rules can significantly improve the accuracy and effectiveness of heuristic-based filtering.

Implementing a denylist approach combined with machine learning to flag potential prompt injection attacks can be a viable and effective option. The denylist would help to catch known harmful prompts, while the machine learning model could potentially identify new, previously unseen threats.

Here's a simple, scalable implementation:

```python
import re

class HeuristicFilter:
    def __init__(self):
        # A list of known harmful prompts
        # Example: "password", "token", "secret"
        self.denylist = ["harmful_prompt1", "harmful_prompt2"]

        # Precompile the regex patterns for efficiency
        self.denylist_patterns = [re.compile(pattern) for pattern in self.denylist]

        # TODO: Initialize machine learning model here
        # self.ml_model = load_model('ml_model.pkl')

    def filter(self, prompt):
        # Denylist check
        for pattern in self.denylist_patterns:
            if pattern.search(prompt):
                return False

        # Machine Learning model check
        # Here we assume the model returns a probability of being malicious
        # If the probability is greater than a threshold, we flag it
        # prob_malicious = self.ml_model.predict([prompt])
        # if prob_malicious > THRESHOLD:
        #     return False

        # If neither the denylist nor the ML model flagged the prompt, we consider it safe
        return True
```

In this code, we initialize the `HeuristicFilter` with a denylist of harmful prompts, and we precompile these into regex patterns for efficiency.
In the `filter` method, we first check if the prompt matches any of the denylist patterns. If it does, we immediately return False, indicating a potentially malicious prompt.

Next, we pass the prompt to a machine learning model, which predicts the likelihood of the prompt being malicious. If this probability exceeds a certain threshold, we also return False. The machine learning model is not implemented in this example, as it would involve considerable additional code and resources.

If the prompt passes both the denylist and machine learning checks, we return True, indicating that it is likely safe. This combination of denylist and machine learning checks provides a robust, scalable solution to prompt injection attacks.

> **Note:** Please note, this is a basic implementation and can be further enhanced to meet specific needs. The machine learning model needs to be trained on a dataset of normal and malicious prompts, which might be a challenging task due to the novelty of prompt injection attacks. However, with sufficient data and regular retraining, this approach could effectively identify and block new types of attacks as they emerge.

## Resources

- https://github.com/NVIDIA/NeMo-Guardrails
- Langchain Prompt Injection Webinar https://www.youtube.com/watch?v=fP6vRNkNEt0
- https://blog.langchain.dev/rebuff/
- Not prompt injection but this is Semantic Similarity Search in Langchain https://python.langchain.com/en/latest/_modules/langchain/prompts/example_selector/semantic_similarity.html
- https://simonwillison.net/2022/Sep/12/prompt-injection/
- https://research.nccgroup.com/2022/12/05/exploring-prompt-injection-attacks/
