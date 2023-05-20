# This is semantic_similarity_check.py

class SemanticSimilarityCheck:
    def __init__(self):
        pass

    def check(self, input, output):
        # Implement your semantic similarity check here.
        pass# secure_prompt/secure_prompt/plugins/semantic_similarity_check.py
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSimilarityCheck:
    def __init__(self, threshold=0.8):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator()
        self.intent_responses = IntentResponses()

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def is_similar(self, text1, text2):
        vec1 = self.model.encode([text1])[0]
        vec2 = self.model.encode([text2])[0]
        return self.cosine_similarity(vec1, vec2) > self.threshold

    def check(self, input, output):
        intent = self.intent_classifier.classify(input)
        typical_responses = self.intent_responses.get_responses(intent)

        for typical_response in typical_responses:
            if self.is_similar(output, typical_response):
                return True

        print("Suspicious output detected")
        return False

# These classes can be in separate files, but I've included them here for brevity
class IntentClassifier:
    def classify(self, text):
        return 'user data request'

class ResponseGenerator:
    def generate(self, intent):
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

