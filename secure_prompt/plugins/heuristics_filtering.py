# secure_prompt/secure_prompt/plugins/heuristics_filtering.py
import re

class HeuristicsFiltering:
    def __init__(self):
        self.denylist = ["harmful_prompt1", "harmful_prompt2"]
        self.denylist_patterns = [re.compile(pattern) for pattern in self.denylist]
        # TODO: Initialize machine learning model here
        # self.ml_model = load_model('ml_model.pkl')

    def filter(self, input):
        for pattern in self.denylist_patterns:
            if pattern.search(input):
                return False

        # Machine Learning model check
        # Here we assume the model returns a probability of being malicious
        # If the probability is greater than a threshold, we flag it
        # prob_malicious = self.ml_model.predict([input])
        # if prob_malicious > THRESHOLD:
        #     return False

        return True
