from typing import Dict, List, Any
from ..services.agent_orchestrator import agent_orchestrator
from ..services.retrieval_service import retrieval_service
from ..models.query import QueryRequest


class SkillsService:
    """
    Service that orchestrates various skills using agents with contextual information
    from the RAG system.
    """

    def __init__(self):
        self.agent_orchestrator = agent_orchestrator
        self.retrieval_service = retrieval_service

    async def execute_summarization(self, query: str, context_text: str = "") -> Dict[str, Any]:
        """
        Execute summarization skill.
        """
        # If no context provided, retrieve relevant chunks from the RAG system
        if not context_text:
            # Create a temporary query request to retrieve context
            temp_request = QueryRequest(query=query)
            source_chunks = self.retrieval_service.retrieve_chunks_by_request(temp_request)

            if source_chunks:
                context_parts = []
                for chunk in source_chunks:
                    content = self.retrieval_service.get_chunk_content(chunk.chunk_id)
                    if content:
                        context_parts.append(content)
                context_text = "\n\n".join(context_parts)

        return await self.agent_orchestrator.execute_agent_task(
            "summarizer",
            query,
            context=context_text
        )

    async def execute_translation(self, query: str, target_language: str = "ur", context_text: str = "") -> Dict[str, Any]:
        """
        Execute translation skill.
        """
        return await self.agent_orchestrator.execute_agent_task(
            "translator",
            query,
            context=context_text,
            target_language=target_language
        )

    async def execute_code_validation(self, code: str) -> Dict[str, Any]:
        """
        Execute ROS code validation skill.
        """
        return await self.agent_orchestrator.execute_agent_task(
            "code_validator",
            "Validate this ROS code",
            context=code
        )

    async def execute_simulation_guidance(self, query: str, context_text: str = "") -> Dict[str, Any]:
        """
        Execute simulation guidance skill.
        """
        # If no context provided, retrieve relevant chunks from the RAG system
        if not context_text:
            temp_request = QueryRequest(query=query)
            source_chunks = self.retrieval_service.retrieve_chunks_by_request(temp_request)

            if source_chunks:
                context_parts = []
                for chunk in source_chunks:
                    content = self.retrieval_service.get_chunk_content(chunk.chunk_id)
                    if content:
                        context_parts.append(content)
                context_text = "\n\n".join(context_parts)

        return await self.agent_orchestrator.execute_agent_task(
            "simulation_helper",
            query,
            context=context_text
        )

    async def execute_multi_skill_task(self, query: str, skills_sequence: List[str] = None) -> Dict[str, Any]:
        """
        Execute a multi-skill task using a sequence of skills.
        """
        if skills_sequence is None:
            # Default sequence based on query type
            skills_sequence = ["summarizer"]  # Example default

        # First, get relevant context from the RAG system
        temp_request = QueryRequest(query=query)
        source_chunks = self.retrieval_service.retrieve_chunks_by_request(temp_request)

        context_text = ""
        if source_chunks:
            context_parts = []
            for chunk in source_chunks:
                content = self.retrieval_service.get_chunk_content(chunk.chunk_id)
                if content:
                    context_parts.append(content)
            context_text = "\n\n".join(context_parts)

        # Execute the sequence of skills
        results = []
        current_context = context_text

        for skill_name in skills_sequence:
            if skill_name == "summarizer":
                result = await self.execute_summarization(query, current_context)
            elif skill_name == "translator":
                result = await self.execute_translation(query, context_text=current_context)
            elif skill_name == "code_validator":
                result = await self.execute_code_validation(query)
            elif skill_name == "simulation_helper":
                result = await self.execute_simulation_guidance(query, current_context)
            else:
                result = {
                    "status": "error",
                    "error": f"Unknown skill: {skill_name}",
                    "agent": skill_name
                }

            results.append(result)

            # Update context with the result for the next skill
            if result["status"] == "success":
                current_context += f"\n\nPrevious result: {result['result']}"

        return {
            "status": "success",
            "results": results,
            "final_context": current_context
        }


# Singleton instance
skills_service = SkillsService()