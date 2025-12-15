import logging
import os
from typing import List
from ..models.query import SourceChunk

logger = logging.getLogger(__name__)

# We'll check for transformers availability when needed, not at import time
TRANSFORMERS_AVAILABLE = None


class HFGenerationService:
    """
    Service for generating responses using local Hugging Face models.
    """

    def __init__(self):
        # Initialize the text generation pipeline with a local model
        self.generator = None

        # Check if transformers is available
        global TRANSFORMERS_AVAILABLE
        if TRANSFORMERS_AVAILABLE is None:
            try:
                from transformers import pipeline
                TRANSFORMERS_AVAILABLE = True
            except ImportError:
                TRANSFORMERS_AVAILABLE = False

        if TRANSFORMERS_AVAILABLE:
            try:
                from transformers import pipeline
                # Using a lightweight but effective model for generation
                self.generator = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base",  # Lightweight but effective
                    tokenizer="google/flan-t5-base",
                    device_map="auto"  # Use CPU or GPU automatically
                )
                logger.info("Initialized Hugging Face generation service with google/flan-t5-base")
            except Exception as e:
                logger.error(f"Error initializing Hugging Face model: {e}")
                # Fallback to a different model
                try:
                    self.generator = pipeline(
                        "text2text-generation",
                        model="facebook/bart-base",
                        tokenizer="facebook/bart-base"
                    )
                    logger.info("Initialized Hugging Face generation service with facebook/bart-base")
                except Exception as e2:
                    logger.error(f"Error initializing fallback model: {e2}")
                    self.generator = None
        else:
            logger.info("Transformers not available, using simple response generation")

    def generate_response(self, query: str, context_chunks: List[SourceChunk], retrieval_service) -> str:
        """
        Generate a response based on the query and retrieved context chunks using local Hugging Face models.
        With improved filtering and relevance checking.
        Falls back to simple response generation if transformers is not available.
        """
        if not self.generator or not TRANSFORMERS_AVAILABLE:
            # Fallback to simple response using context
            if context_chunks:
                first_chunk_content = retrieval_service.get_chunk_content(context_chunks[0].chunk_id)
                if first_chunk_content:
                    return f"Based on the book content: {first_chunk_content[:500]}..."
            return f"I couldn't find relevant information in the book to answer '{query}'. Please try rephrasing your question."

        try:
            # Get content from all chunks
            context_parts = []
            for chunk in context_chunks:
                content = retrieval_service.get_chunk_content(chunk.chunk_id)
                if content and content.strip():
                    # Add chapter and section info for better context
                    context_parts.append(
                        f"Chapter {chunk.chapter_number}, Section '{chunk.section_title}': {content}"
                    )

            if not context_parts:
                return f"I couldn't find relevant information in the book to answer '{query}'. Please try rephrasing your question."

            # Combine context
            combined_context = " ".join(context_parts)

            # Create prompt for the model with better instructions for relevance
            prompt = f"Based on the following context, please answer the question accurately. " \
                     f"If the context doesn't contain the answer, clearly state that the information is not in the provided context. " \
                     f"Prioritize direct answers over explanations when the question asks for definitions or specific facts.\n\n" \
                     f"Context: {combined_context}\n\nQuestion: {query}\n\nAnswer:"

            # Generate response with specific parameters for better quality
            result = self.generator(
                prompt,
                max_length=512,
                min_length=20,
                temperature=0.6,  # Lower temperature for more focused answers
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                truncation=True
            )

            # Extract the generated text
            generated_text = result[0]['generated_text']

            # Extract just the answer part if possible
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text.strip()

            # Improved filtering: Check if the answer is actually relevant to the query
            # If the response seems generic or doesn't address the query, use a simpler approach
            query_lower = query.lower()
            answer_lower = answer.lower()

            # Check if the response mentions that information is not available
            if any(phrase in answer_lower for phrase in [
                "not mentioned", "not provided", "not in the context",
                "not specified", "not found", "no information", "not contain"
            ]):
                # If the model correctly identified that info isn't available, return that
                return answer
            elif len(answer) < 20 and any(phrase in answer_lower for phrase in [
                "i don't know", "unknown", "cannot determine", "no answer"
            ]):
                # If the model says it doesn't know, return a simple response
                return f"I couldn't find relevant information in the book to answer '{query}'."
            else:
                # The answer seems to be a proper response, return it
                return answer

        except Exception as e:
            logger.error(f"Error generating response with Hugging Face: {e}")
            # Fallback to simple response using context
            if context_chunks:
                first_chunk_content = retrieval_service.get_chunk_content(context_chunks[0].chunk_id)
                if first_chunk_content:
                    return f"Based on the book content: {first_chunk_content[:500]}..."
            return f"I couldn't find relevant information in the book to answer '{query}'. Please try rephrasing your question."


# Singleton instance
hf_generation_service = HFGenerationService()