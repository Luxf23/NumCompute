"""
pipeline.py

Simple data processing pipeline.
"""


class Pipeline:
    def __init__(self, steps):
        """
        steps: list of (name, function)
        """
        self.steps = steps

    def run(self, data):
        result = data
        for name, func in self.steps:
            result = func(result)
        return result