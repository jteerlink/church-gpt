"""
Integration tests for Training Pipeline
Tests end-to-end processing from documents to training data.
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from src.data_processing.training_pipeline import (
    TrainingDataPipeline, create_training_pipeline,
    process_conference_content_to_training_data
)


class TestTrainingPipelineIntegration:
    """Integration tests for complete training pipeline."""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample conference documents for testing."""
        return [
            {
                'content': '''Faith is not only a feeling; it is a decision. Faith is choosing to do what the Lord has asked us to do.

When we choose to act with faith, the Lord sees our effort. He sees our heart and blesses us accordingly.

Brothers and sisters, faith is a principle of action and power that changes lives.''',
                'author': 'Dieter F. Uchtdorf',
                'title': 'Acting on Our Faith',
                'date': '2020-04-01',
                'source': 'test-conference/2020-04/acting-on-our-faith.txt'
            },
            {
                'content': '''I know that Jesus Christ lives. I testify that He is our Savior and Redeemer.

My testimony has grown through personal experience and the witness of the Holy Ghost. I bear this witness in the name of Jesus Christ, amen.''',
                'author': 'Russell M. Nelson',
                'title': 'My Testimony',
                'date': '2020-10-01',
                'source': 'test-conference/2020-10/my-testimony.txt'
            }
        ]
    
    def test_end_to_end_pipeline(self, sample_documents):
        """Test complete pipeline from documents to training examples."""
        pipeline = create_training_pipeline(
            tokenizer_name="bert-base-uncased",
            max_tokens=150,
            chunking_strategy="paragraph_aware"
        )
        
        training_examples = pipeline.process_documents_to_training_data(sample_documents)
        
        # Verify training examples generated
        assert len(training_examples) >= 2
        
        # Verify structure of training examples
        for example in training_examples:
            assert 'prompt' in example
            assert 'response' in example
            assert 'metadata' in example
            
            # Verify metadata completeness
            metadata = example['metadata']
            assert 'author' in metadata
            assert 'title' in metadata
            assert 'content_type' in metadata
            assert 'token_count' in metadata
            
            # Verify prompt quality
            assert len(example['prompt']) > 50
            assert metadata['author'] in example['prompt']
            assert metadata['title'] in example['prompt']
    
    def test_content_type_classification_accuracy(self, sample_documents):
        """Test that content types are classified correctly."""
        pipeline = create_training_pipeline(tokenizer_name="bert-base-uncased")
        training_examples = pipeline.process_documents_to_training_data(sample_documents)
        
        # Find examples by author to verify classification
        uchtdorf_examples = [ex for ex in training_examples if ex['metadata']['author'] == 'Dieter F. Uchtdorf']
        nelson_examples = [ex for ex in training_examples if ex['metadata']['author'] == 'Russell M. Nelson']
        
        # Uchtdorf's content should be classified as doctrinal/sermon
        assert any(ex['metadata']['content_type'] in ['doctrinal', 'sermon'] for ex in uchtdorf_examples)
        
        # Nelson's testimony content should be classified as testimony
        assert any(ex['metadata']['content_type'] == 'testimony' for ex in nelson_examples)
    
    def test_pipeline_report_generation(self, sample_documents):
        """Test comprehensive pipeline report generation."""
        pipeline = create_training_pipeline(tokenizer_name="bert-base-uncased")
        training_examples = pipeline.process_documents_to_training_data(sample_documents)
        
        report = pipeline.generate_pipeline_report(sample_documents, training_examples)
        
        # Verify report structure
        assert 'pipeline_summary' in report
        assert 'chunking_quality' in report
        assert 'prompt_analysis' in report
        assert 'quality_indicators' in report
        
        # Verify pipeline summary
        summary = report['pipeline_summary']
        assert summary['input_documents'] == 2
        assert summary['training_examples_generated'] >= 2
        assert summary['expansion_ratio'] >= 1.0
        
        # Verify quality indicators
        quality = report['quality_indicators']
        assert 0 <= quality['chunking_boundary_preservation'] <= 1
        assert 0 <= quality['token_utilization'] <= 1
        assert 0 <= quality['prompt_template_diversity'] <= 1
    
    def test_training_data_saving(self, sample_documents):
        """Test saving training data to file."""
        pipeline = create_training_pipeline(tokenizer_name="bert-base-uncased")
        training_examples = pipeline.process_documents_to_training_data(sample_documents)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test JSONL format
            pipeline.save_training_data(training_examples, temp_path, format="jsonl")
            
            # Verify file was created and contains valid data
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == len(training_examples)
                
                # Verify each line is valid JSON
                for line in lines:
                    data = json.loads(line.strip())
                    assert 'prompt' in data
                    assert 'response' in data
                    assert 'metadata' in data
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_json_format_saving(self, sample_documents):
        """Test saving training data in JSON format."""
        pipeline = create_training_pipeline(tokenizer_name="bert-base-uncased")
        training_examples = pipeline.process_documents_to_training_data(sample_documents)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test JSON format
            pipeline.save_training_data(training_examples, temp_path, format="json")
            
            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
                assert isinstance(data, list)
                assert len(data) == len(training_examples)
                
                for item in data:
                    assert 'prompt' in item
                    assert 'response' in item
                    assert 'metadata' in item
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestPipelineFunctions:
    """Test standalone pipeline functions."""
    
    def test_create_training_pipeline_function(self):
        """Test factory function for creating pipeline."""
        pipeline = create_training_pipeline(
            tokenizer_name="bert-base-uncased",
            max_tokens=100,
            chunking_strategy="sentences"
        )
        
        assert isinstance(pipeline, TrainingDataPipeline)
        assert pipeline.chunker.max_tokens == 100
        assert pipeline.chunking_strategy == "sentences"
    
    def test_process_conference_content_function(self):
        """Test complete processing function."""
        documents = [
            {
                'content': 'Test content for processing.',
                'author': 'Test Author',
                'title': 'Test Title',
                'date': '2024-01-01'
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            report = process_conference_content_to_training_data(
                documents,
                temp_path,
                tokenizer_name="bert-base-uncased",
                max_tokens=100
            )
            
            # Verify report generated
            assert 'pipeline_summary' in report
            assert report['pipeline_summary']['input_documents'] == 1
            
            # Verify file created
            assert os.path.exists(temp_path)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestPipelineQuality:
    """Test pipeline quality and performance characteristics."""
    
    def test_chunking_quality_preservation(self):
        """Test that pipeline preserves chunking quality."""
        documents = [
            {
                'content': '''First paragraph with complete sentences. This has proper punctuation and structure.

Second paragraph continues the thought. It also maintains proper sentence boundaries and formatting.

Third paragraph concludes the document. Final thoughts are presented clearly.''',
                'author': 'Test Author',
                'title': 'Quality Test Document',
                'date': '2024-01-01'
            }
        ]
        
        pipeline = create_training_pipeline(
            tokenizer_name="bert-base-uncased",
            max_tokens=100
        )
        
        training_examples = pipeline.process_documents_to_training_data(documents)
        report = pipeline.generate_pipeline_report(documents, training_examples)
        
        # Verify high boundary preservation
        boundary_preservation = report['quality_indicators']['chunking_boundary_preservation']
        assert boundary_preservation >= 0.8
    
    def test_prompt_template_diversity(self):
        """Test that different content types get different templates."""
        documents = [
            {
                'content': 'Brothers and sisters, let me teach you about faith.',
                'author': 'Author 1',
                'title': 'Teaching About Faith',
                'date': '2024-01-01'
            },
            {
                'content': 'I know that God lives. I testify of His love.',
                'author': 'Author 2', 
                'title': 'My Testimony of God',
                'date': '2024-01-01'
            },
            {
                'content': 'Dear Heavenly Father, we thank thee for this day.',
                'author': 'Author 3',
                'title': 'Opening Prayer',
                'date': '2024-01-01'
            }
        ]
        
        pipeline = create_training_pipeline(tokenizer_name="bert-base-uncased")
        training_examples = pipeline.process_documents_to_training_data(documents)
        
        # Should detect different content types
        content_types = set(ex['metadata']['content_type'] for ex in training_examples)
        assert len(content_types) >= 2  # Should have multiple types
    
    def test_large_document_processing(self):
        """Test pipeline performance with larger documents."""
        # Create larger document
        large_content = "\n\n".join([
            f"This is paragraph {i} with meaningful content about gospel principles. "
            f"It contains multiple sentences and proper structure. "
            f"The content is designed to test chunking and prompt generation quality."
            for i in range(1, 11)
        ])
        
        documents = [
            {
                'content': large_content,
                'author': 'Large Document Author',
                'title': 'Large Document Test',
                'date': '2024-01-01'
            }
        ]
        
        pipeline = create_training_pipeline(
            tokenizer_name="bert-base-uncased",
            max_tokens=200
        )
        
        training_examples = pipeline.process_documents_to_training_data(documents)
        
        # Should create multiple training examples (adjust expectation)
        assert len(training_examples) >= 2
        
        # All examples should respect token limits
        for example in training_examples:
            assert example['metadata']['token_count'] <= 200
            
        # Generate report for quality assessment
        report = pipeline.generate_pipeline_report(documents, training_examples)
        assert report['pipeline_summary']['expansion_ratio'] >= 2.0