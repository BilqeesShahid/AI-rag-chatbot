import os
import re
from typing import List, Dict, Any
from pathlib import Path
import markdown
from markdown import Markdown

from ..models.document import DocumentProcessingRequest, DocumentChunkRequest, ProcessedDocumentChunk, DocumentProcessingResult
from ..utils.helpers import split_text_by_sentences, extract_chapter_info_from_path


def clean_chunk_text(text: str) -> str:
    """
    Clean chunk text by removing YAML frontmatter, excessive whitespace, and other meta content.
    """
    # Remove YAML frontmatter (---something---)
    text = re.sub(r"---[\s\S]*?---", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DocumentProcessor:
    """
    Service for processing documents for the RAG system.
    Handles reading, parsing, and chunking documents from the docs/ folder.
    """

    def __init__(self):
        self.md = Markdown(extensions=['meta', 'fenced_code', 'codehilite', 'tables', 'toc'])

    def process_document_request(self, request: DocumentProcessingRequest) -> DocumentProcessingResult:
        """
        Process a document processing request.
        """
        start_time = __import__('time').time()

        try:
            processed_chunks = self.process_documents_from_path(
                request.source_path,
                request.chunk_size,
                request.overlap
            )

            processing_time = __import__('time').time() - start_time

            return DocumentProcessingResult(
                processed_chunks=processed_chunks,
                total_chunks=len(processed_chunks),
                source_path=request.source_path,
                processing_time=processing_time,
                status="success"
            )
        except Exception as e:
            processing_time = __import__('time').time() - start_time
            return DocumentProcessingResult(
                processed_chunks=[],
                total_chunks=0,
                source_path=request.source_path,
                processing_time=processing_time,
                status=f"error: {str(e)}"
            )

    def process_documents_from_path(self, path: str, chunk_size: int = 400, overlap: int = 50) -> List[ProcessedDocumentChunk]:
        """
        Process documents from a given path (file or directory).
        """
        all_chunks = []

        path_obj = Path(path)

        if path_obj.is_file():
            # Process single file
            chunks = self.process_single_file(path_obj, chunk_size, overlap)
            all_chunks.extend(chunks)
        elif path_obj.is_dir():
            # Process all markdown files in directory recursively
            for file_path in path_obj.rglob("*.md"):
                chunks = self.process_single_file(file_path, chunk_size, overlap)
                all_chunks.extend(chunks)
        else:
            raise ValueError(f"Path does not exist: {path}")

        return all_chunks

    def process_single_file(self, file_path: Path, chunk_size: int = 400, overlap: int = 50) -> List[ProcessedDocumentChunk]:
        """
        Process a single markdown file into chunks.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Convert markdown to plain text
        plain_text = self.md_to_text(content)

        # Get chapter info from path
        chapter_number, section_title = extract_chapter_info_from_path(str(file_path))

        # Chunk the content
        chunks = self.chunk_content(plain_text, chunk_size, overlap, str(file_path), chapter_number, section_title)

        return chunks

    def md_to_text(self, md_content: str) -> str:
        """
        Convert markdown content to plain text.
        """
        # Remove code blocks and their content
        text = re.sub(r'```.*?```', '', md_content, flags=re.DOTALL)
        # Remove inline code
        text = re.sub(r'`(.*?)`', r'\1', text)
        # Remove markdown links but keep the text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Remove image references
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
        # Remove headers but keep the text
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # Remove bold and italic formatting
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
        text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)

        # Clean the text by removing YAML frontmatter and excessive whitespace
        text = clean_chunk_text(text)

        return text.strip()

    def chunk_content(self, content: str, chunk_size: int, overlap: int,
                     source_file_path: str, chapter_number: int, section_title: str) -> List[ProcessedDocumentChunk]:
        """
        Chunk content into smaller pieces.
        """
        # First, try to split by paragraphs
        paragraphs = content.split('\n\n')

        chunks = []
        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed chunk size, start a new chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                # Add the current chunk to the list
                chunks.append(
                    ProcessedDocumentChunk(
                        content=current_chunk.strip(),
                        metadata={
                            "source_file_path": source_file_path,
                            "chapter_number": chapter_number,
                            "section_title": section_title,
                            "chunk_index": chunk_index
                        },
                        chunk_index=chunk_index
                    )
                )

                # Start a new chunk, potentially with overlap
                if overlap > 0 and len(paragraph) > overlap:
                    # Take the last 'overlap' characters from the previous chunk as overlap
                    overlap_content = current_chunk[-overlap:] if len(current_chunk) >= overlap else current_chunk
                    current_chunk = overlap_content + " " + paragraph
                else:
                    current_chunk = paragraph
                chunk_index += 1
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(
                ProcessedDocumentChunk(
                    content=current_chunk.strip(),
                    metadata={
                        "source_file_path": source_file_path,
                        "chapter_number": chapter_number,
                        "section_title": section_title,
                        "chunk_index": chunk_index
                    },
                    chunk_index=chunk_index
                )
            )

        # If chunks are still too large, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk.content) > chunk_size:
                # Split by sentences
                sentence_chunks = split_text_by_sentences(chunk.content, chunk_size)
                for i, sentence_chunk in enumerate(sentence_chunks):
                    final_chunks.append(
                        ProcessedDocumentChunk(
                            content=sentence_chunk,
                            metadata={
                                **chunk.metadata,
                                "chunk_index": f"{chunk.metadata['chunk_index']}.{i}"
                            },
                            chunk_index=f"{chunk.chunk_index}.{i}"
                        )
                    )
            else:
                final_chunks.append(chunk)

        return final_chunks

    def process_chunk_request(self, request: DocumentChunkRequest) -> List[ProcessedDocumentChunk]:
        """
        Process a chunking request for specific content.
        """
        return self.chunk_content(
            request.content,
            request.chunk_size,
            request.overlap,
            request.source_file_path or "unknown",
            request.chapter_number or 0,
            request.section_title or "unknown"
        )


# Singleton instance
document_processor = DocumentProcessor()