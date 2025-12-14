import os
from typing import Optional
from google.generativeai import configure, GenerativeModel
import google.generativeai as genai


class PersonalizationService:
    def __init__(self):
        # Configure the Google Generative AI with API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for personalization")
        configure(api_key=api_key)
        self.model = GenerativeModel('gemini-pro')

    def personalize_content(
        self,
        content: str,
        software_background: str = "beginner",
        hardware_background: str = "beginner",
        ai_skill: str = "beginner"
    ) -> str:
        """
        Personalize content based on user background using Google Generative AI.
        Falls back to simple personalization if AI model is not available.
        """
        try:
            # Create a prompt for the AI model to personalize content
            prompt = f"""
            Please personalize the following content based on the user's background:

            Software Background: {software_background}
            Hardware Background: {hardware_background}
            AI Skill Level: {ai_skill}

            Content to personalize:
            {content}

            Instructions:
            - If the user is a beginner, simplify complex concepts and add more explanations
            - If the user has intermediate experience, provide balanced explanations
            - If the user is advanced, include more technical details and assume prior knowledge
            - Keep the personalized content in markdown format
            - Maintain the core information while adapting the complexity level
            - Add appropriate examples based on the user's background
            - Make sure to preserve the original structure and formatting of the content
            """

            # Generate content using the Gemini model (using synchronous call)
            response = self.model.generate_content(prompt)

            if response and response.text:
                return response.text.strip()
            else:
                # If generation fails, return a simple personalized version
                return self._simple_personalize(content, software_background, hardware_background, ai_skill)

        except Exception as e:
            # If AI personalization fails, return a simple personalized version
            print(f"Error in personalization: {e}")
            return self._simple_personalize(content, software_background, hardware_background, ai_skill)

    def _simple_personalize(self, content: str, software_background: str, hardware_background: str, ai_skill: str) -> str:
        """
        Simple fallback personalization that adds a header based on user background.
        """
        header = f"<!-- Content personalized for: Software={software_background}, Hardware={hardware_background}, AI={ai_skill} -->\n\n"
        return header + content


# Create a singleton instance
personalization_service = PersonalizationService()