"""
Integrated Training Pipeline combining Smart Chunking and Enhanced Prompts
Provides end-to-end processing from raw documents to training-ready prompts.
"""

from typing import List, Dict, Any, Optional
from .chunking import SmartChunker, process_documents_batch
from .prompt_engineering import EnhancedPromptEngine, create_training_prompt, analyze_prompt_effectiveness


class TrainingDataPipeline:
    """Integrated pipeline for processing conference content into training data."""
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased", 
                 max_tokens: int = 512, chunking_strategy: str = "paragraph_aware"):
        """
        Initialize training pipeline.
        
        Args:
            tokenizer_name: Tokenizer to use for chunking
            max_tokens: Maximum tokens per chunk
            chunking_strategy: Strategy for chunking (sentences, paragraph_aware, overlap)
        """
        self.chunker = SmartChunker(tokenizer_name, max_tokens)
        self.prompt_engine = EnhancedPromptEngine()
        self.chunking_strategy = chunking_strategy
    
    def process_documents_to_training_data(self, documents: List[Dict[str, Any]], 
                                         template_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process raw documents into training-ready prompt-response pairs.
        
        Args:
            documents: List of raw documents with metadata
            template_name: Optional specific template to use for all chunks
        
        Returns:
            List of training examples with prompts and responses
        """
        # Step 1: Chunk documents using smart chunking
        chunk_documents, chunking_report = process_documents_batch(
            documents,
            tokenizer_name=self.chunker.tokenizer.name_or_path if hasattr(self.chunker.tokenizer, 'name_or_path') else "unknown",
            max_tokens=self.chunker.max_tokens,
            strategy=self.chunking_strategy
        )
        
        # Step 2: Generate prompts for each chunk
        training_examples = []
        
        for chunk_doc in chunk_documents:
            # Create document metadata for prompt generation
            source_document = {
                'author': chunk_doc['author'],
                'title': chunk_doc['title'],
                'date': chunk_doc['date'],
                'source': chunk_doc['source']
            }
            
            # Generate training prompt
            prompt = self.prompt_engine.generate_prompt(
                source_document, 
                chunk_doc['content'], 
                template_name
            )
            
            # Create training example
            training_example = {
                'prompt': prompt,
                'response': chunk_doc['content'],
                'metadata': {
                    'author': chunk_doc['author'],
                    'title': chunk_doc['title'],
                    'date': chunk_doc['date'],
                    'source': chunk_doc['source'],
                    'chunk_id': chunk_doc['chunk_id'],
                    'total_chunks': chunk_doc['total_chunks'],
                    'token_count': chunk_doc['token_count'],
                    'content_type': self.prompt_engine.content_classifier.classify_content(
                        chunk_doc['title'], 
                        chunk_doc['content']
                    ).value,
                    'template_used': template_name or 'auto_selected'
                }
            }
            
            training_examples.append(training_example)
        
        return training_examples
    
    def generate_pipeline_report(self, documents: List[Dict[str, Any]], 
                               training_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive pipeline processing report.
        
        Args:
            documents: Original input documents
            training_examples: Generated training examples
        
        Returns:
            Comprehensive report on pipeline processing
        """
        # Chunking analysis
        chunk_documents = [
            {
                'content': example['response'],
                'token_count': example['metadata']['token_count']
            }
            for example in training_examples
        ]
        
        chunking_report = self.chunker.generate_chunking_report(documents, chunk_documents)
        
        # Prompt analysis
        prompt_analysis = analyze_prompt_effectiveness(documents)
        
        # Template usage analysis
        template_usage = {}
        content_type_distribution = {}
        
        for example in training_examples:
            template = example['metadata']['template_used']
            content_type = example['metadata']['content_type']
            
            template_usage[template] = template_usage.get(template, 0) + 1
            content_type_distribution[content_type] = content_type_distribution.get(content_type, 0) + 1
        
        # Quality metrics
        avg_prompt_length = sum(len(ex['prompt']) for ex in training_examples) / len(training_examples)
        avg_response_length = sum(len(ex['response']) for ex in training_examples) / len(training_examples)
        
        return {
            'pipeline_summary': {
                'input_documents': len(documents),
                'training_examples_generated': len(training_examples),
                'expansion_ratio': len(training_examples) / len(documents) if documents else 0,
                'avg_prompt_length': avg_prompt_length,
                'avg_response_length': avg_response_length
            },
            'chunking_quality': chunking_report,
            'prompt_analysis': {
                'template_usage': template_usage,
                'content_type_distribution': content_type_distribution,
                'template_diversity': len(template_usage),
                'content_type_coverage': len(content_type_distribution)
            },
            'quality_indicators': {
                'chunking_boundary_preservation': chunking_report.get('boundary_analysis', {}).get('sentence_preservation_rate', 0),
                'token_utilization': chunking_report.get('quality_metrics', {}).get('token_utilization', 0),
                'prompt_template_diversity': len(template_usage) / len(training_examples) if training_examples else 0
            }
        }
    
    def save_training_data(self, training_examples: List[Dict[str, Any]], 
                          output_path: str, format: str = "jsonl") -> None:
        """
        Save training examples to file.
        
        Args:
            training_examples: Generated training examples
            output_path: Path to save training data
            format: Output format (jsonl, json)
        """
        import json
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in training_examples:
                    json.dump(example, f, ensure_ascii=False)
                    f.write('\n')
        elif format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_examples, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


def create_training_pipeline(tokenizer_name: str = "bert-base-uncased",
                           max_tokens: int = 512,
                           chunking_strategy: str = "paragraph_aware") -> TrainingDataPipeline:
    """
    Factory function to create configured training pipeline.
    
    Args:
        tokenizer_name: Tokenizer for chunking
        max_tokens: Maximum tokens per chunk
        chunking_strategy: Chunking strategy to use
    
    Returns:
        Configured TrainingDataPipeline instance
    """
    return TrainingDataPipeline(tokenizer_name, max_tokens, chunking_strategy)


def process_conference_content_to_training_data(
    input_documents: List[Dict[str, Any]],
    output_path: str,
    tokenizer_name: str = "google/gemma-7b",
    max_tokens: int = 512,
    chunking_strategy: str = "paragraph_aware",
    template_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete pipeline function to convert conference documents to training data.
    
    Args:
        input_documents: Raw conference documents
        output_path: Path to save training data
        tokenizer_name: Tokenizer for chunking
        max_tokens: Maximum tokens per chunk
        chunking_strategy: Chunking strategy
        template_name: Optional specific template
    
    Returns:
        Processing report with quality metrics
    """
    # Create pipeline
    pipeline = TrainingDataPipeline(tokenizer_name, max_tokens, chunking_strategy)
    
    # Process documents
    training_examples = pipeline.process_documents_to_training_data(
        input_documents, 
        template_name
    )
    
    # Save training data
    pipeline.save_training_data(training_examples, output_path)
    
    # Generate report
    report = pipeline.generate_pipeline_report(input_documents, training_examples)
    
    return report