"""
Unit tests for Enhanced Prompt Engineering System
Tests content classification, template selection, and prompt generation.
"""

import pytest
from src.data_processing.prompt_engineering import (
    EnhancedPromptEngine, ContentClassifier, ContentType, PromptTemplate,
    create_training_prompt, analyze_prompt_effectiveness
)


class TestContentClassifier:
    """Test suite for content type classification."""
    
    @pytest.fixture
    def classifier(self):
        """Create content classifier instance."""
        return ContentClassifier()
    
    def test_sermon_classification(self, classifier):
        """Test classification of sermon content."""
        title = "The Gospel of Jesus Christ"
        content = """Brothers and sisters, I want to speak today about the fundamental principles of the gospel. Let me share with you three important truths that will help us understand our purpose in life."""
        
        content_type = classifier.classify_content(title, content)
        assert content_type == ContentType.SERMON
    
    def test_testimony_classification(self, classifier):
        """Test classification of testimony content."""
        title = "My Testimony of the Savior"
        content = """I know that Jesus Christ lives. I testify that He is our Savior and Redeemer. My testimony has grown through personal experience and the witness of the Holy Ghost."""
        
        content_type = classifier.classify_content(title, content)
        assert content_type == ContentType.TESTIMONY
    
    def test_prayer_classification(self, classifier):
        """Test classification of prayer content."""
        title = "Opening Prayer"
        content = """Our dear Heavenly Father, we thank Thee for this opportunity to gather together. We pray that Thy Spirit will be with us as we learn and grow. In the name of Jesus Christ, amen."""
        
        content_type = classifier.classify_content(title, content)
        assert content_type == ContentType.PRAYER
    
    def test_teaching_classification(self, classifier):
        """Test classification of teaching content."""
        title = "Learning Through Scripture Study"
        content = """We learn from the scriptures that consistent study brings understanding. Let us understand how to make scripture study more meaningful in our daily lives."""
        
        content_type = classifier.classify_content(title, content)
        assert content_type == ContentType.TEACHING
    
    def test_doctrinal_classification(self, classifier):
        """Test classification of doctrinal content."""
        title = "The Doctrine of Christ"
        content = """The doctrine of Christ is clear and eternal. This principle teaches us about the nature of God and our relationship with Him."""
        
        content_type = classifier.classify_content(title, content)
        assert content_type == ContentType.DOCTRINAL
    
    def test_personal_story_classification(self, classifier):
        """Test classification of personal story content."""
        title = "Lessons I Learned from My Father"
        content = """I remember when I was a young boy, my father taught me an important lesson. Years ago, this experience taught me about the power of faith."""
        
        content_type = classifier.classify_content(title, content)
        assert content_type == ContentType.PERSONAL_STORY
    
    def test_fallback_classification(self, classifier):
        """Test fallback to GENERAL for unclassifiable content."""
        title = "Random Title"
        content = "Random content without specific patterns."
        
        content_type = classifier.classify_content(title, content)
        assert content_type == ContentType.GENERAL


class TestPromptTemplate:
    """Test suite for PromptTemplate class."""
    
    def test_template_creation(self):
        """Test template creation and basic functionality."""
        template = PromptTemplate(
            name="test_template",
            template="Hello {name}, this is about {topic}.",
            description="Test template",
            content_types=[ContentType.GENERAL]
        )
        
        assert template.name == "test_template"
        assert template.description == "Test template"
        assert ContentType.GENERAL in template.content_types
    
    def test_template_formatting(self):
        """Test template formatting with variables."""
        template = PromptTemplate(
            name="test",
            template="Author: {author}, Title: {title}",
            description="Test",
            content_types=[ContentType.GENERAL]
        )
        
        formatted = template.format(author="Test Author", title="Test Title")
        assert "Author: Test Author" in formatted
        assert "Title: Test Title" in formatted


class TestEnhancedPromptEngine:
    """Test suite for EnhancedPromptEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create prompt engine instance."""
        return EnhancedPromptEngine()
    
    @pytest.fixture
    def sample_document(self):
        """Sample document for testing."""
        return {
            'author': 'Russell M. Nelson',
            'title': 'Let God Prevail',
            'date': '2020-10-01',
            'content': 'Full conference talk content here...'
        }
    
    def test_engine_initialization(self, engine):
        """Test that engine initializes with templates."""
        assert len(engine.templates) >= 5  # Should have multiple templates
        assert "style_mimicry" in engine.templates
        assert "completion" in engine.templates
        assert "testimony" in engine.templates
        assert "prayer" in engine.templates
    
    def test_prompt_generation_with_auto_classification(self, engine, sample_document):
        """Test automatic prompt generation with content classification."""
        chunk_content = "Brothers and sisters, let me teach you about faith and testimony."
        
        prompt = engine.generate_prompt(sample_document, chunk_content)
        
        assert "Russell M. Nelson" in prompt
        assert "Let God Prevail" in prompt
        assert "2020-10-01" in prompt
        assert len(prompt) > 100  # Should be substantial prompt
    
    def test_prompt_generation_with_specific_template(self, engine, sample_document):
        """Test prompt generation with specified template."""
        chunk_content = "I know that Jesus Christ lives and loves us."
        
        prompt = engine.generate_prompt(sample_document, chunk_content, template_name="testimony")
        
        assert "testimony" in prompt.lower()
        assert "Russell M. Nelson" in prompt
        assert len(prompt) > 50
    
    def test_template_selection_for_content_types(self, engine):
        """Test that appropriate templates are selected for content types."""
        assert engine._select_template_for_content_type(ContentType.SERMON) == "style_mimicry"
        assert engine._select_template_for_content_type(ContentType.TESTIMONY) == "testimony"
        assert engine._select_template_for_content_type(ContentType.PRAYER) == "prayer"
        assert engine._select_template_for_content_type(ContentType.TEACHING) == "style_mimicry"
    
    def test_topic_extraction(self, engine):
        """Test main topic extraction from content."""
        faith_content = "Faith is the foundation of all righteousness. We must have faith and trust in the Lord."
        topic = engine._extract_main_topic(faith_content)
        assert topic == "faith"
        
        family_content = "The family is central to the plan of salvation. Parents and children have sacred responsibilities."
        topic = engine._extract_main_topic(family_content)
        assert topic == "family"
        
        # Test fallback for unrecognized content
        random_content = "Random text without specific gospel themes."
        topic = engine._extract_main_topic(random_content)
        assert topic == "gospel principles"


class TestPromptFunctions:
    """Test standalone prompt functions."""
    
    def test_create_training_prompt_function(self):
        """Test standalone prompt creation function."""
        document = {
            'author': 'Dieter F. Uchtdorf',
            'title': 'Acting on Our Faith',
            'date': '2020-04-01'
        }
        
        chunk_content = "Faith is not only a feeling; it is a decision."
        prompt = create_training_prompt(document, chunk_content)
        
        assert "Dieter F. Uchtdorf" in prompt
        assert "Acting on Our Faith" in prompt
        assert len(prompt) > 50
    
    def test_analyze_prompt_effectiveness(self):
        """Test prompt effectiveness analysis."""
        documents = [
            {
                'title': 'The Power of Faith',
                'content': 'Brothers and sisters, faith is essential.',
                'author': 'Test Author 1'
            },
            {
                'title': 'My Testimony',
                'content': 'I know that God lives and loves us.',
                'author': 'Test Author 2'
            },
            {
                'title': 'Opening Prayer',
                'content': 'Dear Heavenly Father, we thank thee.',
                'author': 'Test Author 3'
            }
        ]
        
        analysis = analyze_prompt_effectiveness(documents)
        
        assert 'content_type_distribution' in analysis
        assert 'template_usage' in analysis
        assert analysis['total_documents'] == 3
        assert analysis['classification_coverage'] >= 1
        assert analysis['template_diversity'] >= 1


class TestPromptIntegration:
    """Test prompt system integration scenarios."""
    
    def test_multiple_content_types_handling(self):
        """Test handling multiple content types in batch."""
        engine = EnhancedPromptEngine()
        
        test_cases = [
            {
                'doc': {'title': 'Faith and Works', 'author': 'Author 1', 'date': '2024-01-01'},
                'content': 'Brothers and sisters, let me teach about faith.',
                'expected_type': ContentType.SERMON
            },
            {
                'doc': {'title': 'My Witness', 'author': 'Author 2', 'date': '2024-01-01'},
                'content': 'I testify that Jesus Christ lives.',
                'expected_type': ContentType.TESTIMONY
            },
            {
                'doc': {'title': 'Benediction', 'author': 'Author 3', 'date': '2024-01-01'},
                'content': 'Heavenly Father, we thank thee for this day.',
                'expected_type': ContentType.PRAYER
            }
        ]
        
        for case in test_cases:
            content_type = engine.content_classifier.classify_content(
                case['doc']['title'], 
                case['content']
            )
            prompt = engine.generate_prompt(case['doc'], case['content'])
            
            # Verify classification
            assert content_type == case['expected_type']
            # Verify prompt generation
            assert case['doc']['author'] in prompt
            assert len(prompt) > 50
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        engine = EnhancedPromptEngine()
        
        # Empty document
        empty_doc = {}
        prompt = engine.generate_prompt(empty_doc, "Some content")
        assert "Unknown Author" in prompt
        assert "Untitled" in prompt
        
        # Missing fields
        partial_doc = {'author': 'Test Author'}
        prompt = engine.generate_prompt(partial_doc, "Test content")
        assert "Test Author" in prompt
        assert "Untitled" in prompt
        assert "Unknown Date" in prompt
    
    def test_prompt_length_validation(self):
        """Test that generated prompts are reasonable length."""
        engine = EnhancedPromptEngine()
        document = {
            'author': 'Test Author',
            'title': 'Test Title',
            'date': '2024-01-01'
        }
        
        short_content = "Short content."
        long_content = "Very long content. " * 100
        
        short_prompt = engine.generate_prompt(document, short_content)
        long_prompt = engine.generate_prompt(document, long_content)
        
        # Both should be reasonable length
        assert 50 < len(short_prompt) < 1000
        assert 50 < len(long_prompt) < 1000