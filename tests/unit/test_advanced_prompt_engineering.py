"""
Unit tests for Advanced Prompt Engineering System (Phase 4)
Tests speaker adaptation, difficulty progression, and conversation modes.
"""

import pytest
from src.data_processing.advanced_prompt_engineering import (
    AdvancedPromptEngine, SpeakerProfileManager, PromptValidator,
    AdvancedPromptTemplate, ContentType, DifficultyLevel, PromptMode,
    SpeakerProfile, create_advanced_training_prompt, analyze_prompt_effectiveness
)


class TestSpeakerProfileManager:
    """Test suite for speaker profile management."""
    
    def test_profile_creation(self):
        """Test creation of speaker profiles."""
        manager = SpeakerProfileManager()
        
        # Verify profiles exist
        assert "Russell M. Nelson" in manager.profiles
        assert "Dieter F. Uchtdorf" in manager.profiles
        assert "Jeffrey R. Holland" in manager.profiles
        
        # Test profile retrieval
        nelson_profile = manager.get_profile("Russell M. Nelson")
        assert nelson_profile is not None
        assert nelson_profile.name == "Russell M. Nelson"
        assert nelson_profile.difficulty_preference == DifficultyLevel.COMPLEX
        assert len(nelson_profile.common_phrases) > 0
    
    def test_speaker_context_creation(self):
        """Test speaker context dictionary creation."""
        manager = SpeakerProfileManager()
        
        context = manager.create_speaker_context("Dieter F. Uchtdorf")
        assert 'speaking_style' in context
        assert 'common_phrases' in context
        assert 'rhetorical_patterns' in context
        assert 'topic_expertise' in context
        
        # Test unknown speaker fallback
        unknown_context = manager.create_speaker_context("Unknown Speaker")
        assert unknown_context['speaking_style'] == 'authoritative and caring'
    
    def test_get_all_speakers(self):
        """Test getting all available speakers."""
        manager = SpeakerProfileManager()
        speakers = manager.get_all_speakers()
        
        assert len(speakers) >= 3
        assert "Russell M. Nelson" in speakers


class TestAdvancedPromptTemplate:
    """Test suite for advanced prompt templates."""
    
    def test_template_creation(self):
        """Test template creation with all parameters."""
        template = AdvancedPromptTemplate(
            name="test_template",
            template="Write as {author} about {topic}",
            description="Test template",
            content_types=[ContentType.SERMON],
            difficulty_level=DifficultyLevel.MODERATE,
            prompt_mode=PromptMode.COMPLETION,
            requires_context=True
        )
        
        assert template.name == "test_template"
        assert template.difficulty_level == DifficultyLevel.MODERATE
        assert template.requires_context == True
        assert template.get_complexity_score() == 0.5
    
    def test_template_formatting(self):
        """Test template formatting with variables."""
        template = AdvancedPromptTemplate(
            name="format_test",
            template="Hello {author}, please write about {topic}",
            description="Format test",
            content_types=[ContentType.GENERAL]
        )
        
        result = template.format(author="Test Author", topic="faith")
        assert "Test Author" in result
        assert "faith" in result
    
    def test_template_formatting_with_missing_keys(self):
        """Test template formatting handles missing keys gracefully."""
        template = AdvancedPromptTemplate(
            name="missing_key_test",
            template="Write as {author} about {missing_key}",
            description="Missing key test",
            content_types=[ContentType.GENERAL]
        )
        
        result = template.format(author="Test Author")
        assert "Test Author" in result
        assert "[missing_key]" in result or "Missing template variable" in result
    
    def test_complexity_scores(self):
        """Test complexity score calculation."""
        simple_template = AdvancedPromptTemplate(
            name="simple", template="test", description="test",
            content_types=[ContentType.GENERAL],
            difficulty_level=DifficultyLevel.SIMPLE
        )
        assert simple_template.get_complexity_score() == 0.25
        
        advanced_template = AdvancedPromptTemplate(
            name="advanced", template="test", description="test",
            content_types=[ContentType.GENERAL],
            difficulty_level=DifficultyLevel.ADVANCED
        )
        assert advanced_template.get_complexity_score() == 1.0


class TestPromptValidator:
    """Test suite for prompt validation."""
    
    @pytest.fixture
    def validator(self):
        return PromptValidator()
    
    @pytest.fixture
    def valid_template(self):
        return AdvancedPromptTemplate(
            name="valid_template",
            template="Write a passage in {author}'s style about faith and testimony.",
            description="Valid test template",
            content_types=[ContentType.SERMON]
        )
    
    def test_template_validation_success(self, validator, valid_template):
        """Test successful template validation."""
        result = validator.validate_template(valid_template)
        
        assert result['is_valid'] == True
        assert len(result['issues']) == 0
        assert 'complexity_score' in result
    
    def test_template_validation_too_short(self, validator):
        """Test validation failure for short template."""
        short_template = AdvancedPromptTemplate(
            name="short", template="Hi", description="test",
            content_types=[ContentType.GENERAL]
        )
        
        result = validator.validate_template(short_template)
        assert result['is_valid'] == False
        assert any("too short" in issue.lower() for issue in result['issues'])
    
    def test_template_validation_missing_author(self, validator):
        """Test validation failure for missing author variable."""
        no_author_template = AdvancedPromptTemplate(
            name="no_author", 
            template="Write a passage about faith and testimony in someone's style.",
            description="test",
            content_types=[ContentType.GENERAL]
        )
        
        result = validator.validate_template(no_author_template)
        assert result['is_valid'] == False
        assert any("author" in issue.lower() for issue in result['issues'])
    
    def test_generated_prompt_validation_success(self, validator):
        """Test successful generated prompt validation."""
        prompt = "Write a passage in Elder Holland's style about faith and testimony."
        
        result = validator.validate_generated_prompt(prompt)
        assert result['is_valid'] == True
        assert result['length'] > 0
        assert result['estimated_tokens'] > 0
    
    def test_generated_prompt_validation_unfilled_vars(self, validator):
        """Test validation failure for unfilled variables."""
        prompt = "Write a passage in {author}'s style about {topic}."
        
        result = validator.validate_generated_prompt(prompt)
        assert result['is_valid'] == False
        assert any("unfilled" in issue.lower() for issue in result['issues'])


class TestAdvancedPromptEngine:
    """Test suite for advanced prompt engine."""
    
    @pytest.fixture
    def engine(self):
        return AdvancedPromptEngine()
    
    @pytest.fixture
    def sample_document(self):
        return {
            'author': 'Russell M. Nelson',
            'title': 'Revelation and the Church',
            'date': '2024-04-01',
            'content': 'I testify that revelation continues in our day.'
        }
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert len(engine.templates) > 0
        assert engine.speaker_manager is not None
        assert engine.prompt_validator is not None
        assert engine.content_classifier is not None
    
    def test_template_creation(self, engine):
        """Test that all expected templates are created."""
        expected_templates = [
            "simple_completion", "conversation", "speaker_adapted",
            "interview", "instruction_following", "testimony", "prayer"
        ]
        
        for template_name in expected_templates:
            assert template_name in engine.templates
            assert isinstance(engine.templates[template_name], AdvancedPromptTemplate)
    
    def test_content_classification(self, engine):
        """Test content classification."""
        # Test testimony classification
        testimony_type = engine.content_classifier.classify_content(
            "My Testimony", "I know that Jesus Christ lives"
        )
        assert testimony_type == ContentType.TESTIMONY
        
        # Test prayer classification
        prayer_type = engine.content_classifier.classify_content(
            "Opening Prayer", "Our dear Heavenly Father, we thank thee"
        )
        assert prayer_type == ContentType.PRAYER
    
    def test_topic_extraction(self, engine):
        """Test topic extraction from content."""
        faith_content = "Faith is a principle of action and power that helps us believe"
        topic = engine._extract_main_topic(faith_content)
        assert topic == "faith"
        
        family_content = "Family is central to our Heavenly Father's plan for our eternal happiness"
        topic = engine._extract_main_topic(family_content)
        assert topic == "family"
    
    def test_template_selection(self, engine, sample_document):
        """Test optimal template selection."""
        # Test with known speaker
        template_name = engine._select_optimal_template(
            sample_document, "I testify that revelation continues", DifficultyLevel.COMPLEX
        )
        assert template_name in engine.templates
        
        # Test template has appropriate difficulty level
        template = engine.templates[template_name]
        assert template.difficulty_level == DifficultyLevel.COMPLEX or template_name == "simple_completion"
    
    def test_template_variable_preparation(self, engine, sample_document):
        """Test template variable preparation with speaker profiles."""
        speaker_profile = engine.speaker_manager.get_profile("Russell M. Nelson")
        
        variables = engine._prepare_template_variables(
            sample_document, "Test content", speaker_profile, "Test question"
        )
        
        # Check required variables
        assert 'author' in variables
        assert 'topic' in variables
        assert 'speaking_style' in variables
        assert 'common_phrases' in variables
        assert 'question' in variables
        
        # Check speaker-specific content
        assert variables['author'] == 'Russell M. Nelson'
        assert variables['speaking_style'] == speaker_profile.speaking_style
    
    def test_advanced_prompt_generation(self, engine, sample_document):
        """Test advanced prompt generation."""
        prompt = engine.generate_advanced_prompt(
            sample_document,
            "I testify that revelation continues in our day.",
            template_name="testimony"
        )
        
        assert len(prompt) > 50  # Reasonable length
        assert "Russell M. Nelson" in prompt
        assert "{" not in prompt or "}" not in prompt  # No unfilled variables
    
    def test_conversation_prompt_creation(self, engine):
        """Test conversation prompt creation."""
        prompt = engine.create_conversation_prompt(
            "Dieter F. Uchtdorf",
            "How can I develop more faith?",
            "faith and spiritual growth"
        )
        
        assert "Dieter F. Uchtdorf" in prompt
        assert "How can I develop more faith?" in prompt
        assert len(prompt) > 50
    
    def test_difficulty_progression(self, engine):
        """Test difficulty progression functionality."""
        simple_templates = engine.get_difficulty_progression(DifficultyLevel.SIMPLE)
        advanced_templates = engine.get_difficulty_progression(DifficultyLevel.ADVANCED)
        
        assert len(simple_templates) > 0
        assert len(advanced_templates) > 0
        assert simple_templates != advanced_templates
    
    def test_template_validation_all(self, engine):
        """Test validation of all templates."""
        validation_results = engine.validate_all_templates()
        
        assert len(validation_results) == len(engine.templates)
        for template_name, result in validation_results.items():
            assert 'is_valid' in result
            assert 'issues' in result


class TestSpeakerAdaptation:
    """Test suite for speaker-specific adaptations."""
    
    @pytest.fixture
    def engine(self):
        return AdvancedPromptEngine()
    
    def test_nelson_adaptation(self, engine):
        """Test adaptation for Russell M. Nelson's style."""
        document = {
            'author': 'Russell M. Nelson',
            'title': 'Prophetic Guidance',
            'date': '2024-04-01'
        }
        
        prompt = engine.generate_advanced_prompt(
            document,
            "The Lord provides revelation through His prophets.",
            template_name="speaker_adapted"
        )
        
        # Should include Nelson's characteristics
        nelson_profile = engine.speaker_manager.get_profile("Russell M. Nelson")
        assert any(phrase in prompt for phrase in nelson_profile.common_phrases)
        assert nelson_profile.speaking_style in prompt
    
    def test_uchtdorf_adaptation(self, engine):
        """Test adaptation for Dieter F. Uchtdorf's style."""
        document = {
            'author': 'Dieter F. Uchtdorf',
            'title': 'Faith and Hope',
            'date': '2024-04-01'
        }
        
        prompt = engine.generate_advanced_prompt(
            document,
            "Faith gives us hope for the future.",
            template_name="speaker_adapted"
        )
        
        # Should include Uchtdorf's characteristics
        uchtdorf_profile = engine.speaker_manager.get_profile("Dieter F. Uchtdorf")
        assert uchtdorf_profile.speaking_style in prompt
    
    def test_unknown_speaker_fallback(self, engine):
        """Test fallback for unknown speakers."""
        document = {
            'author': 'Unknown Speaker',
            'title': 'General Message',
            'date': '2024-04-01'
        }
        
        prompt = engine.generate_advanced_prompt(
            document,
            "Gospel principles guide our lives.",
            template_name="speaker_adapted"
        )
        
        # Should use default characteristics
        assert "authoritative and caring" in prompt or "Unknown Speaker" in prompt


class TestConversationModes:
    """Test suite for conversation mode functionality."""
    
    @pytest.fixture
    def engine(self):
        return AdvancedPromptEngine()
    
    def test_conversation_template(self, engine):
        """Test conversation template functionality."""
        document = {
            'author': 'Jeffrey R. Holland',
            'title': 'Conversation',
            'date': '2024-04-01'
        }
        
        prompt = engine.generate_advanced_prompt(
            document,
            "Discussion about faith",
            template_name="conversation",
            conversation_context="How do I strengthen my faith?"
        )
        
        assert "Jeffrey R. Holland" in prompt
        assert "How do I strengthen my faith?" in prompt
        assert ":" in prompt  # Conversation format
    
    def test_interview_template(self, engine):
        """Test interview template functionality."""
        document = {
            'author': 'Russell M. Nelson',
            'title': 'Interview',
            'date': '2024-04-01'
        }
        
        # First, let's directly test the template
        template = engine.templates["interview"]
        template_vars = engine._prepare_template_variables(
            document, 
            "Interview about revelation", 
            engine.speaker_manager.get_profile("Russell M. Nelson"),
            "What is your counsel about receiving revelation?"
        )
        direct_prompt = template.format(**template_vars)
        
        # Test that direct template formatting works
        assert "Russell M. Nelson" in direct_prompt
        assert "What is your counsel about receiving revelation?" in direct_prompt
        
        # Now test through the engine
        prompt = engine.generate_advanced_prompt(
            document,
            "Interview about revelation",
            template_name="interview",
            conversation_context="What is your counsel about receiving revelation?"
        )
        
        assert "Russell M. Nelson" in prompt
        # The prompt should contain the question, either directly or through validation fallback
        assert ("What is your counsel about receiving revelation?" in prompt or 
                "Interview about revelation" in prompt)


class TestDifficultyProgression:
    """Test suite for difficulty progression."""
    
    @pytest.fixture
    def engine(self):
        return AdvancedPromptEngine()
    
    def test_difficulty_level_selection(self, engine):
        """Test that difficulty levels are respected in template selection."""
        document = {
            'author': 'Jeffrey R. Holland',  # Advanced preference
            'title': 'Complex Teaching',
            'date': '2024-04-01'
        }
        
        # Should prefer advanced templates for Holland
        advanced_prompt = engine.generate_advanced_prompt(
            document,
            "Complex doctrinal content",
            difficulty_level=DifficultyLevel.ADVANCED
        )
        
        simple_prompt = engine.generate_advanced_prompt(
            document,
            "Simple faith message",
            difficulty_level=DifficultyLevel.SIMPLE
        )
        
        # Advanced prompt should be more complex
        assert len(advanced_prompt) >= len(simple_prompt)
    
    def test_speaker_difficulty_preference(self, engine):
        """Test that speaker difficulty preferences are used."""
        # Uchtdorf prefers moderate difficulty
        uchtdorf_doc = {
            'author': 'Dieter F. Uchtdorf',
            'title': 'Faith Message',
            'date': '2024-04-01'
        }
        
        # Holland prefers advanced difficulty
        holland_doc = {
            'author': 'Jeffrey R. Holland',
            'title': 'Faith Message',
            'date': '2024-04-01'
        }
        
        uchtdorf_template = engine._select_optimal_template(uchtdorf_doc, "faith content", None)
        holland_template = engine._select_optimal_template(holland_doc, "faith content", None)
        
        uchtdorf_complexity = engine.templates[uchtdorf_template].get_complexity_score()
        holland_complexity = engine.templates[holland_template].get_complexity_score()
        
        # Holland should get more complex templates
        assert holland_complexity >= uchtdorf_complexity


class TestFactoryFunctions:
    """Test suite for factory functions."""
    
    def test_create_advanced_training_prompt(self):
        """Test advanced training prompt creation function."""
        document = {
            'author': 'Russell M. Nelson',
            'title': 'Revelation',
            'date': '2024-04-01'
        }
        
        prompt = create_advanced_training_prompt(
            document,
            "Revelation comes to those who seek it.",
            template_name="testimony"
        )
        
        assert len(prompt) > 50
        assert "Russell M. Nelson" in prompt
    
    def test_analyze_prompt_effectiveness(self):
        """Test prompt effectiveness analysis."""
        documents = [
            {
                'author': 'Russell M. Nelson',
                'title': 'My Testimony',
                'content': 'I know that Jesus Christ lives and guides His church.'
            },
            {
                'author': 'Dieter F. Uchtdorf',
                'title': 'Faith and Hope',
                'content': 'Faith gives us hope and strength in difficult times.'
            },
            {
                'author': 'Unknown Author',
                'title': 'Prayer',
                'content': 'Our Heavenly Father, we thank thee for thy blessings.'
            }
        ]
        
        analysis = analyze_prompt_effectiveness(documents)
        
        # Check analysis structure
        assert 'content_type_distribution' in analysis
        assert 'template_usage' in analysis
        assert 'difficulty_distribution' in analysis
        assert 'speaker_coverage' in analysis
        assert 'total_documents' in analysis
        assert analysis['total_documents'] == 3
        
        # Check that supported speakers are identified
        assert analysis['supported_speakers'] >= 2  # Nelson and Uchtdorf


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def engine(self):
        return AdvancedPromptEngine()
    
    def test_complete_workflow(self, engine):
        """Test complete workflow from document to validated prompt."""
        document = {
            'author': 'Dieter F. Uchtdorf',
            'title': 'The Power of Faith',
            'date': '2024-04-01',
            'content': 'Faith is a principle of action and power.'
        }
        
        # Generate prompt
        prompt = engine.generate_advanced_prompt(
            document,
            "Faith helps us overcome obstacles and grow closer to God.",
            difficulty_level=DifficultyLevel.MODERATE
        )
        
        # Validate prompt
        validation = engine.prompt_validator.validate_generated_prompt(prompt)
        
        assert validation['is_valid']
        assert len(validation['issues']) == 0
        assert validation['length'] > 50
        assert validation['estimated_tokens'] > 10
    
    def test_multiple_template_types(self, engine):
        """Test different template types work correctly."""
        document = {
            'author': 'Jeffrey R. Holland',
            'title': 'Faith and Testimony',
            'date': '2024-04-01'
        }
        
        content = "I bear witness that Jesus Christ lives and loves each of us."
        
        # Test different template types
        templates_to_test = ["testimony", "conversation", "speaker_adapted"]
        
        for template_name in templates_to_test:
            prompt = engine.generate_advanced_prompt(
                document, content, template_name=template_name
            )
            
            assert len(prompt) > 50
            assert "Jeffrey R. Holland" in prompt
            validation = engine.prompt_validator.validate_generated_prompt(prompt)
            assert validation['is_valid'], f"Template {template_name} failed validation: {validation['issues']}"