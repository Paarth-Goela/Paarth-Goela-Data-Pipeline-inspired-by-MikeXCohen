#!/usr/bin/env python3
"""
Simple AI Assistant for Neural Data Analysis
Provides helpful responses without external API dependencies
"""

import random
import re

class SimpleAIAssistant:
    """A simple AI assistant for neural data analysis questions"""
    
    def __init__(self):
        self.responses = {
            'greeting': [
                "Hello! I'm your neural data analysis assistant. How can I help you today?",
                "Hi there! I'm here to help with your EEG/LFP analysis. What would you like to know?",
                "Welcome! I can help you with neural data preprocessing, analysis, and interpretation."
            ],
            'preprocessing': [
                "For preprocessing, I recommend: 1) Filtering (1-40 Hz bandpass), 2) Artifact removal with ICA, 3) Bad channel detection, 4) Re-referencing to average. Start with the sample data to see these steps in action!",
                "Key preprocessing steps: Filter out line noise (50/60 Hz), remove eye blinks and muscle artifacts with ICA, detect and mark bad channels, and create clean epochs. The app has built-in tools for all of these.",
                "Preprocessing pipeline: 1) High-pass filter at 1 Hz, 2) Low-pass filter at 40 Hz, 3) Notch filter at 50/60 Hz, 4) ICA for artifact removal, 5) Epoch creation with baseline correction."
            ],
            'analysis': [
                "You can run time-frequency analysis, PAC, ERP, and connectivity analysis from the sidebar. Each module has tooltips and help text.",
                "For PAC, try the Tort method for robust results. For connectivity, coherence and PLV are most common.",
                "Use the batch processing feature to analyze multiple files at once."
            ],
            'error': [
                "If you encounter an error, check that your data is properly formatted and all required fields are filled.",
                "Common errors include missing events, incompatible file types, or not enough epochs after preprocessing.",
                "If a plot doesn't show, try re-running the previous step or check the logs for details."
            ],
            'goodbye': [
                "Good luck with your analysis! Let me know if you have more questions.",
                "Happy analyzing! 🧠",
                "Feel free to ask anything else about neural data analysis."
            ]
        }

    def get_response(self, user_input):
        user_input = user_input.lower()
        if any(word in user_input for word in ['hello', 'hi', 'hey']):
            return random.choice(self.responses['greeting'])
        if 'preprocess' in user_input or 'filter' in user_input or 'ica' in user_input:
            return random.choice(self.responses['preprocessing'])
        if 'analy' in user_input or 'pac' in user_input or 'erp' in user_input or 'connect' in user_input:
            return random.choice(self.responses['analysis'])
        if 'error' in user_input or 'fail' in user_input or 'not working' in user_input:
            return random.choice(self.responses['error'])
        if any(word in user_input for word in ['bye', 'goodbye', 'exit']):
            return random.choice(self.responses['goodbye'])
        return "I'm here to help with neural data analysis. Please ask a specific question or select a module to get started."

def ai_assistant(user_input):
    assistant = SimpleAIAssistant()
    return assistant.get_response(user_input) 