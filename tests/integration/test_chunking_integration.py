"""
Integration tests for Smart Chunking System
Tests chunking with real General Conference content and validates quality improvements.
"""

import pytest
import os
from pathlib import Path
from src.data_processing.chunking import SmartChunker, process_documents_batch


class TestChunkingIntegration:
    """Integration tests using real General Conference content."""
    
    @pytest.fixture
    def sample_conference_content(self):
        """Load sample conference content for testing."""
        return {
            'content': '''Faith is not only a feeling; it is a decision. Faith is choosing to do what the Lord has asked us to do. When we choose to act with faith—especially when we do not feel very faithful—the Lord sees our effort. He sees our heart.

The prophet Alma taught his son Helaman about the power of faith when he said: "And now, my son, I would that ye should understand that these things are not without a shadow; for as our fathers were slothful to give heed unto this compass ... behold, they did not prosper; even so it is with us."

Brothers and sisters, faith is not passive. Faith is not hoping that things will work out. Faith is not wishful thinking. Faith is a principle of action and power.''',
            'author': 'Dieter F. Uchtdorf',
            'title': 'Acting on Our Faith',
            'date': '2020-04-01',
            'source': 'general-conference/2020-04/acting-on-our-faith.txt'
        }
    
    def test_chunking_preserves_meaning(self, sample_conference_content):
        """Test that chunking preserves semantic meaning and context."""
        chunker = SmartChunker("google/gemma-7b", max_tokens=200)
        
        chunk_docs = chunker.process_document(sample_conference_content, strategy="paragraph_aware")
        
        assert len(chunk_docs) >= 2  # Should split into multiple chunks
        
        # Verify metadata preservation
        for chunk_doc in chunk_docs:
            assert chunk_doc['author'] == 'Dieter F. Uchtdorf'
            assert chunk_doc['title'] == 'Acting on Our Faith'
            assert chunk_doc['date'] == '2020-04-01'
        
        # Verify content quality
        combined_content = ' '.join(chunk['content'] for chunk in chunk_docs)
        original_sentences = sample_conference_content['content'].count('.')
        combined_sentences = combined_content.count('.')
        
        # Should preserve most sentences (allowing for some boundary effects)
        assert combined_sentences >= original_sentences * 0.9
    
    def test_chunking_quality_metrics(self, sample_conference_content):
        """Test quality metrics on real content."""
        chunker = SmartChunker("google/gemma-7b", max_tokens=150)
        
        chunks = chunker.chunk_by_paragraphs(sample_conference_content['content'])
        metrics = chunker.calculate_chunk_quality_score(chunks)
        
        # Quality thresholds for conference content
        assert metrics['boundary_preservation'] >= 0.8  # Most chunks should end properly
        assert metrics['token_utilization'] >= 0.5      # Reasonable token usage
        assert metrics['size_consistency'] >= 0.3       # Reasonable consistency
    
    def test_strategy_comparison_on_real_content(self, sample_conference_content):
        """Compare chunking strategies on real conference content."""
        chunker = SmartChunker("google/gemma-7b", max_tokens=200)
        
        sentence_chunks = chunker.chunk_by_sentences(sample_conference_content['content'])
        paragraph_chunks = chunker.chunk_by_paragraphs(sample_conference_content['content'])
        
        sentence_metrics = chunker.calculate_chunk_quality_score(sentence_chunks)
        paragraph_metrics = chunker.calculate_chunk_quality_score(paragraph_chunks)
        
        # Paragraph-aware should generally have better boundary preservation
        assert paragraph_metrics['boundary_preservation'] >= sentence_metrics['boundary_preservation'] * 0.9
    
    @pytest.mark.skipif(not os.path.exists("/Users/jaredteerlink/repos/church-gpt/formatted_content"), 
                       reason="Formatted content directory not found")
    def test_real_conference_file_chunking(self):
        """Test chunking with actual conference file."""
        # Find a sample conference file
        content_dir = Path("/Users/jaredteerlink/repos/church-gpt/formatted_content/general-conference")
        conference_files = list(content_dir.glob("**/*/a-*.txt"))
        
        if not conference_files:
            pytest.skip("No conference files found for testing")
        
        # Read first available file
        sample_file = conference_files[0]
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create test document
        document = {
            'content': content,
            'author': 'Test Author',
            'title': sample_file.stem,
            'date': '2024-01-01',
            'source': str(sample_file)
        }
        
        chunker = SmartChunker("google/gemma-7b", max_tokens=512)
        chunk_docs = chunker.process_document(document, strategy="paragraph_aware")
        
        # Verify chunking worked
        assert len(chunk_docs) > 0
        assert all(chunk['token_count'] <= 512 for chunk in chunk_docs)
        
        # Generate quality report
        report = chunker.generate_chunking_report([document], chunk_docs)
        
        # Verify reasonable quality metrics
        assert report['summary']['token_preservation'] >= 0.95  # Minimal token loss
        assert report['quality_metrics']['token_utilization'] >= 0.4  # Reasonable utilization
        
        print(f"✅ Processed {sample_file.name}: {len(chunk_docs)} chunks, "
              f"{report['quality_metrics']['token_utilization']:.2f} utilization")


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    def test_batch_processing_multiple_documents(self):
        """Test processing multiple documents in batch."""
        documents = [
            {
                'content': 'First document content with multiple sentences. Here is more content.',
                'author': 'Author One',
                'title': 'Title One'
            },
            {
                'content': 'Second document content here. This has different content and structure.',
                'author': 'Author Two',
                'title': 'Title Two'
            }
        ]
        
        chunks, report = process_documents_batch(
            documents,
            tokenizer_name="google/gemma-7b",
            max_tokens=100,
            strategy="paragraph_aware"
        )
        
        assert len(chunks) >= 2  # Should create multiple chunks
        assert report['summary']['original_docs'] == 2
        
        # Verify metadata preservation across documents
        authors = set(chunk['author'] for chunk in chunks)
        assert 'Author One' in authors
        assert 'Author Two' in authors
    
    def test_batch_processing_quality_report(self):
        """Test that batch processing generates comprehensive quality reports."""
        documents = [
            {'content': 'Test content for quality analysis. Multiple sentences here.'}
        ]
        
        chunks, report = process_documents_batch(documents, max_tokens=50)
        
        # Verify report structure
        assert 'summary' in report
        assert 'quality_metrics' in report
        assert 'boundary_analysis' in report
        
        # Verify metrics are calculated
        assert report['quality_metrics']['avg_chunk_size'] > 0
        assert 0 <= report['quality_metrics']['token_utilization'] <= 1
        assert 0 <= report['boundary_analysis']['sentence_preservation_rate'] <= 1


class TestChunkingPerformance:
    """Performance and edge case tests."""
    
    def test_large_document_handling(self):
        """Test chunking performance with large documents."""
        # Create large document
        large_content = " ".join([
            f"This is sentence {i} in a very large document." 
            for i in range(1000)
        ])
        
        document = {
            'content': large_content,
            'author': 'Test Author',
            'title': 'Large Document Test'
        }
        
        chunker = SmartChunker("google/gemma-7b", max_tokens=512)
        chunk_docs = chunker.process_document(document, strategy="paragraph_aware")
        
        # Verify all chunks respect token limits
        assert all(chunk['token_count'] <= 512 for chunk in chunk_docs)
        assert len(chunk_docs) > 1  # Should be split into multiple chunks
    
    def test_edge_cases(self):
        """Test edge cases: empty content, single sentence, no punctuation."""
        chunker = SmartChunker("google/gemma-7b", max_tokens=100)
        
        # Empty content
        empty_doc = {'content': ''}
        chunks = chunker.process_document(empty_doc)
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0]['content'].strip() == '')
        
        # Single sentence
        single_sentence_doc = {'content': 'Single sentence here.'}
        chunks = chunker.process_document(single_sentence_doc)
        assert len(chunks) == 1
        assert chunks[0]['content'] == 'Single sentence here.'
        
        # No punctuation
        no_punct_doc = {'content': 'No punctuation here just words'}
        chunks = chunker.process_document(no_punct_doc)
        assert len(chunks) >= 1