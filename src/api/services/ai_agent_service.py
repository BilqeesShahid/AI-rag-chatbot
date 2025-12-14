import logging
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew
from langchain.tools import tool
from ..services.retrieval_service import retrieval_service
from ..models.query import QueryRequest

logger = logging.getLogger(__name__)


class AIAgentService:
    """
    Service for creating and managing AI agents using CrewAI and OpenAI Agents SDK.
    Integrates with the RAG system to provide intelligent responses to user queries.
    """

    def __init__(self):
        self.retrieval_service = retrieval_service
        self.agent_crew = None

    def search_knowledge_base(self, query: str) -> str:
        """
        Search the knowledge base (Qdrant) for relevant information to answer user queries.
        """
        try:
            # Use the retrieval service to get relevant chunks
            source_chunks = self.retrieval_service.retrieve_chunks(query, top_k=5)

            if not source_chunks:
                return "No relevant information found in the knowledge base."

            # Format the results for the agent
            results = []
            for chunk in source_chunks:
                chunk_content = self.retrieval_service.get_chunk_content(chunk.chunk_id)
                results.append({
                    "chapter": chunk.chapter_number,
                    "section": chunk.section_title,
                    "content": chunk_content[:500] + "..." if len(chunk_content) > 500 else chunk_content,  # Truncate long content
                    "source": chunk.source_file_path
                })

            return str(results)
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return f"Error searching knowledge base: {str(e)}"

    def get_search_tool(self):
        """
        Get the knowledge base search tool in a format compatible with CrewAI.
        """
        from langchain.tools import tool

        @tool("knowledge_base_search")
        def search_knowledge_base_tool(query: str) -> str:
            """
            Search the knowledge base (Qdrant) for relevant information to answer user queries.
            """
            return self.search_knowledge_base(query)

        return search_knowledge_base_tool

    def create_agent_crew(self):
        """
        Create a crew of AI agents for handling book-related queries.
        """
        # Get the properly formatted tool
        search_tool = self.get_search_tool()

        # Create the researcher agent
        researcher_agent = Agent(
            role="Senior Robotics and AI Researcher",
            goal="Find accurate and relevant information from the robotics and AI book to answer user queries",
            backstory="You are an expert researcher with deep knowledge of robotics, AI, and humanoid systems. "
                     "You have access to a comprehensive book on Physical AI & Humanoid Robotics and can search "
                     "through its contents to find relevant information for users.",
            verbose=True,
            tools=[search_tool]
        )

        # Create the response agent
        response_agent = Agent(
            role="Knowledgeable AI Assistant",
            goal="Provide clear, accurate, and helpful responses to users based on information from the robotics and AI book",
            backstory="You are a helpful AI assistant with expertise in robotics and AI. You provide "
                     "detailed, well-structured answers to user questions based on the content from "
                     "the Physical AI & Humanoid Robotics book. Always cite sources when possible.",
            verbose=True
        )

        # Create the crew
        self.agent_crew = Crew(
            agents=[researcher_agent, response_agent],
            tasks=[
                Task(
                    description=f"Search the knowledge base for information related to this query: {{query}}. "
                               f"Find the most relevant chapters and content that can help answer the question.",
                    agent=researcher_agent,
                    expected_output="A summary of relevant information found in the knowledge base, "
                                 "including chapter numbers, sections, and key content that addresses the query."
                ),
                Task(
                    description="Based on the research provided, create a comprehensive and helpful response "
                               "to the user's query. Include relevant citations to chapters and sections "
                               "where the information was found. Make sure the response is clear, accurate, "
                               "and helpful to someone learning about robotics and AI.",
                    agent=response_agent,
                    expected_output="A well-structured response to the user's query that incorporates "
                                 "information from the knowledge base with proper citations to relevant "
                                 "chapters and sections."
                )
            ],
            verbose=2
        )

        return self.agent_crew

    async def get_agent_response(self, query: str) -> str:
        """
        Get a response from the AI agent crew for the given query.
        """
        try:
            if not self.agent_crew:
                self.create_agent_crew()

            # Execute the crew with the user query
            result = self.agent_crew.kickoff({
                "query": query
            })

            return str(result)
        except Exception as e:
            logger.error(f"Error getting agent response: {e}")
            return f"Sorry, I encountered an error processing your request: {str(e)}"


# Singleton instance
ai_agent_service = AIAgentService()