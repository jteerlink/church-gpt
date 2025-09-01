"""
Smart Chunking System for LDS General Conference Content
Implements semantic boundary-aware chunking for improved fine-tuning quality.
"""

import re
import statistics
from typing import List, Dict, Any
from transformers import AutoTokenizer


class SmartChunker:
    """Advanc2ed chunking system that respects semantic boundaries."""
    
    def __init__(self, tokenizer_name: str = "google/gemma-7b", max_tokens: int = 512):
        """Initialize chunker with tokenizer and token limits."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into semantic chunks at sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            token_count = len(self.tokenizer.encode(test_chunk))
            
            if token_count <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """Enhanced chunking with paragraph boundary awareness."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            token_count = len(self.tokenizer.encode(test_chunk))
            
            if token_count <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle oversized paragraphs
                if len(self.tokenizer.encode(paragraph)) > self.max_tokens:
                    para_chunks = self.chunk_by_sentences(paragraph)
                    chunks.extend(para_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_with_overlap(self, text: str, overlap_tokens: int = 50) -> List[str]:
        """Create chunks with controlled overlap for context preservation."""
        base_chunks = self.chunk_by_paragraphs(text)
        overlapped_chunks = []
        
        for i, chunk in enumerate(base_chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_tokens = self.tokenizer.encode(base_chunks[i-1])
                overlap_start = max(0, len(prev_tokens) - overlap_tokens)
                overlap_text = self.tokenizer.decode(prev_tokens[overlap_start:])
                
                combined_chunk = overlap_text + "\n\n" + chunk
                overlapped_chunks.append(combined_chunk)
        
        return overlapped_chunks
    
    def calculate_chunk_quality_score(self, chunks: List[str]) -> Dict[str, float]:
        """Calculate quality metrics for chunking results."""
        if not chunks:
            return {}
            
        metrics = {
            'avg_tokens': 0,
            'token_utilization': 0,
            'boundary_preservation': 0,
            'size_consistency': 0
        }
        
        token_counts = [len(self.tokenizer.encode(chunk)) for chunk in chunks]
        
        metrics['avg_tokens'] = sum(token_counts) / len(token_counts)
        metrics['token_utilization'] = metrics['avg_tokens'] / self.max_tokens
        
        # Boundary preservation: chunks ending with sentence terminators
        complete_chunks = sum(1 for chunk in chunks if chunk.strip()[-1] in '.!?')
        metrics['boundary_preservation'] = complete_chunks / len(chunks) if chunks else 0
        
        # Size consistency: standard deviation of token counts
        if len(token_counts) > 1 and metrics['avg_tokens'] > 0:
            metrics['size_consistency'] = 1 - (statistics.stdev(token_counts) / metrics['avg_tokens'])
        else:
            metrics['size_consistency'] = 1.0
        
        return metrics
    
    def process_document(self, document: Dict[str, Any], 
                        strategy: str = "paragraph_aware") -> List[Dict[str, Any]]:
        """Process a complete document into semantic chunks."""
        text = document.get('content', '')
        
        if strategy == "sentences":
            chunks = self.chunk_by_sentences(text)
        elif strategy == "paragraph_aware":
            chunks = self.chunk_by_paragraphs(text)
        elif strategy == "overlap":
            chunks = self.chunk_with_overlap(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        # Create chunk documents with metadata
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                'content': chunk,
                'author': document.get('author', ''),
                'title': document.get('title', ''),
                'date': document.get('date', ''),
                'source': document.get('source', ''),
                'chunk_id': i,
                'total_chunks': len(chunks),
                'token_count': len(self.tokenizer.encode(chunk))
            }
            chunk_documents.append(chunk_doc)
        
        return chunk_documents
    
    def generate_chunking_report(self, documents: List[Dict], 
                               chunk_documents: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive quality report for chunking results."""
        original_tokens = sum(len(self.tokenizer.encode(doc['content'])) 
                             for doc in documents)
        chunked_tokens = sum(chunk['token_count'] for chunk in chunk_documents)
        
        chunk_sizes = [chunk['token_count'] for chunk in chunk_documents]
        
        report = {
            'summary': {
                'original_docs': len(documents),
                'total_chunks': len(chunk_documents),
                'chunks_per_doc': len(chunk_documents) / len(documents) if documents else 0,
                'token_preservation': chunked_tokens / original_tokens if original_tokens > 0 else 0
            },
            'quality_metrics': {
                'avg_chunk_size': statistics.mean(chunk_sizes) if chunk_sizes else 0,
                'chunk_size_std': statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0,
                'token_utilization': statistics.mean(chunk_sizes) / self.max_tokens if chunk_sizes else 0,
                'size_consistency': 1 - (statistics.stdev(chunk_sizes) / statistics.mean(chunk_sizes)) if len(chunk_sizes) > 1 and statistics.mean(chunk_sizes) > 0 else 1.0
            },
            'boundary_analysis': {
                'complete_sentences': sum(1 for chunk in chunk_documents 
                                        if chunk['content'].strip() and chunk['content'].strip()[-1] in '.!?'),
                'sentence_preservation_rate': sum(1 for chunk in chunk_documents 
                                                if chunk['content'].strip() and chunk['content'].strip()[-1] in '.!?') / len(chunk_documents) if chunk_documents else 0
            }
        }
        
        return report


def process_documents_batch(documents: List[Dict[str, Any]], 
                          tokenizer_name: str = "google/gemma-7b",
                          max_tokens: int = 512,
                          strategy: str = "paragraph_aware") -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process a batch of documents with chunking and quality reporting."""
    chunker = SmartChunker(tokenizer_name, max_tokens)
    
    all_chunks = []
    for document in documents:
        doc_chunks = chunker.process_document(document, strategy)
        all_chunks.extend(doc_chunks)
    
    report = chunker.generate_chunking_report(documents, all_chunks)
    
    return all_chunks, report