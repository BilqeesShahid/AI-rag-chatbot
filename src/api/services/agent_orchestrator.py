import openai
from typing import Dict, List, Any, Optional
from ..utils.constants import GOOGLE_API_KEY
import google.generativeai as genai
import asyncio

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)


class AgentOrchestrator:
    """
    Orchestrates multiple agents for different skills using OpenAI Agents concepts
    but implemented with our existing tools and Gemini for multi-step reasoning.
    """

    def __init__(self):
        self.agents = {
            'summarizer': self._summarize_agent,
            'translator': self._translate_agent,
            'code_validator': self._code_validation_agent,
            'simulation_helper': self._simulation_helper_agent,
        }

    async def execute_agent_task(self, agent_name: str, query: str, context: str = "", **kwargs) -> Dict[str, Any]:
        """
        Execute a task using the specified agent.
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not available")

        try:
            result = await self.agents[agent_name](query, context, **kwargs)
            return {
                "status": "success",
                "result": result,
                "agent": agent_name
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent": agent_name
            }

    async def _summarize_agent(self, query: str, context: str, **kwargs) -> str:
        """
        Agent for summarization tasks.
        """
        prompt = f"""
        Please provide a concise summary of the following text.
        Focus on the key points and main concepts.

        Text to summarize:
        {context or query}

        Summary:
        """

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=500
            )
        )

        return response.text if response.text else "Could not generate summary."

    async def _translate_agent(self, query: str, context: str, target_language: str = "ur", **kwargs) -> str:
        """
        Agent for translation tasks.
        """
        text_to_translate = context or query

        if target_language.lower() == "ur":
            prompt = f"""
            Translate the following text to Urdu (اردو).
            Make sure the translation is accurate and natural-sounding.

            Text to translate:
            {text_to_translate}

            Urdu translation:
            """
        else:
            # Default to English
            prompt = f"""
            Translate the following text to English.
            Make sure the translation is accurate and natural-sounding.

            Text to translate:
            {text_to_translate}

            English translation:
            """

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000
            )
        )

        return response.text if response.text else "Could not generate translation."

    async def _code_validation_agent(self, query: str, context: str, **kwargs) -> str:
        """
        Agent for ROS code validation.
        """
        code_to_validate = context or query

        prompt = f"""
        Analyze the following ROS code for correctness, best practices, and potential issues.
        Provide feedback on:
        1. Syntax correctness
        2. Best practices adherence
        3. Potential runtime issues
        4. Suggestions for improvement

        ROS Code to validate:
        {code_to_validate}

        Analysis and feedback:
        """

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
                max_output_tokens=800
            )
        )

        return response.text if response.text else "Could not analyze code."

    async def _simulation_helper_agent(self, query: str, context: str, **kwargs) -> str:
        """
        Agent for simulation guidance and verification.
        """
        simulation_query = context or query

        prompt = f"""
        Provide guidance on the following simulation query related to robotics.
        Include best practices, common pitfalls, and recommendations.

        Simulation query:
        {simulation_query}

        Guidance and recommendations:
        """

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=800
            )
        )

        return response.text if response.text else "Could not provide simulation guidance."

    async def execute_multi_step_reasoning(self, query: str, context: str = "", agents_sequence: List[str] = None) -> Dict[str, Any]:
        """
        Execute multi-step reasoning using a sequence of agents.
        """
        if agents_sequence is None:
            # Default sequence for complex queries
            agents_sequence = ['summarizer', 'translator']  # Example sequence

        results = []
        current_context = context

        for agent_name in agents_sequence:
            result = await self.execute_agent_task(agent_name, query, current_context)
            results.append(result)

            # Update context with the result for the next agent
            if result["status"] == "success":
                current_context += f"\n\nPrevious result: {result['result']}"

        return {
            "status": "success",
            "results": results,
            "final_context": current_context
        }


# Singleton instance
agent_orchestrator = AgentOrchestrator()