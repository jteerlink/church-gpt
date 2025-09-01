"""
Advanced Prompt Engineering System for LDS General Conference Fine-Tuning
Implements sophisticated prompt templates with speaker adaptation and conversation modes.
"""

import re
import random
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class ContentType(Enum):
    """Enumeration of General Conference content types."""
    SERMON = "sermon"
    TESTIMONY = "testimony"
    PRAYER = "prayer"
    TEACHING = "teaching"
    DOCTRINAL = "doctrinal"
    PERSONAL_STORY = "personal_story"
    CONVERSATION = "conversation"
    INTERVIEW = "interview"
    GENERAL = "general"


class DifficultyLevel(Enum):
    """Training difficulty levels for progressive complexity."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class PromptMode(Enum):
    """Different prompt interaction modes."""
    COMPLETION = "completion"
    CONVERSATION = "conversation"
    INSTRUCTION = "instruction"
    STYLE_TRANSFER = "style_transfer"


@dataclass
class SpeakerProfile:
    """Profile containing speaker-specific characteristics."""
    name: str
    common_phrases: List[str]
    rhetorical_patterns: List[str]
    topic_expertise: List[str]
    speaking_style: str
    difficulty_preference: DifficultyLevel


class PromptTemplate:
    """Container for advanced prompt template configuration."""
    
    def __init__(self, name: str, template: str, description: str, 
                 content_types: List[ContentType],
                 difficulty_level: DifficultyLevel = DifficultyLevel.MODERATE,
                 prompt_mode: PromptMode = PromptMode.COMPLETION,
                 requires_context: bool = False):
        self.name = name
        self.template = template
        self.description = description
        self.content_types = content_types
        self.difficulty_level = difficulty_level
        self.prompt_mode = prompt_mode
        self.requires_context = requires_context
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        return self.template.format(**kwargs)
    
    def get_complexity_score(self) -> float:
        """Calculate template complexity score for training progression."""
        complexity_map = {
            DifficultyLevel.SIMPLE: 0.25,
            DifficultyLevel.MODERATE: 0.5,
            DifficultyLevel.COMPLEX: 0.75,
            DifficultyLevel.ADVANCED: 1.0
        }
        return complexity_map.get(self.difficulty_level, 0.5)


class SpeakerProfileManager:
    """Manages speaker-specific profiles and characteristics."""
    
    def __init__(self):
        """Initialize with predefined speaker profiles."""
        self.profiles = self._create_speaker_profiles()
    
    def _create_speaker_profiles(self) -> Dict[str, SpeakerProfile]:
        """Create speaker profiles for common General Conference speakers."""
        profiles = {}
        
        profiles["Russell M. Nelson"] = SpeakerProfile(
            name="Russell M. Nelson",
            common_phrases=[
                "I know that", "The Lord has taught", "My dear brothers and sisters",
                "I testify that", "May I suggest", "Consider this"
            ],
            rhetorical_patterns=[
                "structured numbered points", "medical analogies", "scriptural citations",
                "personal experiences", "direct challenges"
            ],
            topic_expertise=[
                "revelation", "prophetic guidance", "temple work", "family proclamation"
            ],
            speaking_style="authoritative and caring",
            difficulty_preference=DifficultyLevel.COMPLEX
        )
        
        profiles["Dieter F. Uchtdorf"] = SpeakerProfile(
            name="Dieter F. Uchtdorf",
            common_phrases=[
                "Brothers and sisters", "Let me share", "From my experience",
                "This I know", "May I invite you", "Sometimes we"
            ],
            rhetorical_patterns=[
                "aviation analogies", "storytelling", "gentle questions",
                "personal vulnerability", "universal experiences"
            ],
            topic_expertise=[
                "faith", "hope", "patience", "spiritual growth", "grace"
            ],
            speaking_style="warm and relatable",
            difficulty_preference=DifficultyLevel.MODERATE
        )
        
        profiles["Jeffrey R. Holland"] = SpeakerProfile(
            name="Jeffrey R. Holland",
            common_phrases=[
                "I bear witness", "With all my heart", "My beloved brothers and sisters",
                "I promise you", "Never give up", "Christ loves you"
            ],
            rhetorical_patterns=[
                "emotional appeals", "passionate testimony", "scriptural stories",
                "direct challenges", "personal assurance"
            ],
            topic_expertise=[
                "atonement", "testimony", "discipleship", "trials", "faith"
            ],
            speaking_style="passionate and direct",
            difficulty_preference=DifficultyLevel.ADVANCED
        )
        
        return profiles
    
    def get_profile(self, speaker_name: str) -> Optional[SpeakerProfile]:
        """Get speaker profile by name."""
        return self.profiles.get(speaker_name)
    
    def get_all_speakers(self) -> List[str]:
        """Get list of all available speakers."""
        return list(self.profiles.keys())


class EnhancedPromptEngine:
    """Advanced prompt engineering system for conference content."""
    
    def __init__(self):
        """Initialize prompt engine with advanced templates and speaker profiles."""
        self.templates = self._create_templates()
        self.content_classifier = ContentClassifier()
        self.speaker_manager = SpeakerProfileManager()
        self.prompt_validator = PromptValidator()
    
    def _create_templates(self) -> Dict[str, PromptTemplate]:
        """Create advanced prompt templates with difficulty progression and conversation modes."""
        templates = {}
        
        # Simple completion template
        templates["simple_completion"] = PromptTemplate(
            name="simple_completion",
            template="""Complete the following text in {author}'s style:

"{partial_text}"

Continue in {author}'s voice:

""",
            description="Simple text completion for basic training",
            content_types=[ContentType.GENERAL, ContentType.SERMON],
            difficulty_level=DifficultyLevel.SIMPLE,
            prompt_mode=PromptMode.COMPLETION
        )
        
        # Conversation-style template
        templates["conversation"] = PromptTemplate(
            name="conversation",
            template="""You are {author}. A member approaches you after conference and asks: "{question}"

Respond as {author} would, incorporating their characteristic {speaking_style} approach:

{author}: """,
            description="Conversational response in speaker's style",
            content_types=[ContentType.CONVERSATION, ContentType.TEACHING],
            difficulty_level=DifficultyLevel.MODERATE,
            prompt_mode=PromptMode.CONVERSATION,
            requires_context=True
        )
        
        # Speaker-adapted style mimicry
        templates["speaker_adapted"] = PromptTemplate(
            name="speaker_adapted",
            template="""Write a passage on {topic} in {author}'s distinctive style, incorporating their characteristic phrases: {common_phrases} and rhetorical approach.

Context: Drawing from {author}'s expertise in {topic_expertise} and their {speaking_style} delivery style.

Key characteristics to include:
- Rhetorical patterns: {rhetorical_patterns}
- Speaking style: {speaking_style}

Passage on {topic} in {author}'s voice:

""",
            description="Advanced speaker-specific adaptation",
            content_types=[ContentType.SERMON, ContentType.TEACHING, ContentType.DOCTRINAL],
            difficulty_level=DifficultyLevel.COMPLEX,
            prompt_mode=PromptMode.STYLE_TRANSFER,
            requires_context=True
        )
        
        # Interview-style template
        templates["interview"] = PromptTemplate(
            name="interview",
            template="""In an interview setting, {author} is asked: "{question}"

Drawing from their experience with {topic_expertise} and their {speaking_style} approach, provide {author}'s thoughtful response:

Interviewer: {question}

{author}: """,
            description="Interview-style responses for complex topics",
            content_types=[ContentType.INTERVIEW, ContentType.CONVERSATION],
            difficulty_level=DifficultyLevel.ADVANCED,
            prompt_mode=PromptMode.CONVERSATION,
            requires_context=True
        )
        
        # Advanced instruction following
        templates["instruction_following"] = PromptTemplate(
            name="instruction_following",
            template="""As {author}, provide guidance on {topic} by following this specific structure:

1. Begin with a characteristic opening phrase
2. Share a relevant personal insight or scripture
3. Provide practical application
4. Close with testimony or invitation

Topic: {topic}
Speaking style: {speaking_style}
Use these characteristic elements: {rhetorical_patterns}

{author}'s structured response:

""",
            description="Structured instruction following with speaker adaptation",
            content_types=[ContentType.TEACHING, ContentType.DOCTRINAL],
            difficulty_level=DifficultyLevel.ADVANCED,
            prompt_mode=PromptMode.INSTRUCTION,
            requires_context=True
        )
        
        # Enhanced testimony template
        templates["testimony"] = PromptTemplate(
            name="testimony",
            template="""Share a testimony about {topic} in the style of {author}, reflecting their characteristic way of bearing witness.

Context: Incorporate {author}'s {speaking_style} approach and use elements from: {common_phrases}

Personal testimony about {topic} in {author}'s voice:

""",
            description="Personal testimony with speaker adaptation",
            content_types=[ContentType.TESTIMONY, ContentType.PERSONAL_STORY],
            difficulty_level=DifficultyLevel.MODERATE,
            prompt_mode=PromptMode.STYLE_TRANSFER,
            requires_context=True
        )
        
        # Keep existing prayer template but update it
        templates["prayer"] = PromptTemplate(
            name="prayer",
            template="""Offer a prayer in the reverent style of {author}, reflecting their characteristic approach to communion with Heavenly Father.

Context: Drawing from the spirit of their devotional style and {speaking_style} approach

Prayer in {author}'s devotional style:

""",
            description="Devotional and prayer content with speaker adaptation",
            content_types=[ContentType.PRAYER],
            difficulty_level=DifficultyLevel.MODERATE,
            prompt_mode=PromptMode.STYLE_TRANSFER
        )
        
        return templates


class PromptValidator:
    """Validates prompt templates and generated prompts for quality and consistency."""
    
    def __init__(self):
        """Initialize validator with quality criteria."""
        self.min_prompt_length = 50
        self.max_prompt_length = 2000
        self.required_elements = {
            'has_clear_instruction': True,
            'includes_context': True,
            'specifies_output_format': True
        }
    
    def validate_template(self, template: PromptTemplate) -> Dict[str, Any]:
        """Validate template structure and completeness."""
        issues = []
        
        # Check template length
        if len(template.template) < self.min_prompt_length:
            issues.append("Template too short")
        elif len(template.template) > self.max_prompt_length:
            issues.append("Template too long")
        
        # Check for required placeholders
        required_vars = ['{author}']
        for var in required_vars:
            if var not in template.template:
                issues.append(f"Missing required variable: {var}")
        
        # Check for clear instruction
        instruction_indicators = ['write', 'generate', 'complete', 'respond', 'share', 'offer']
        if not any(indicator in template.template.lower() for indicator in instruction_indicators):
            issues.append("Template lacks clear instruction")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'complexity_score': template.get_complexity_score()
        }
    
    def validate_generated_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate generated prompt for training suitability."""
        issues = []
        
        # Check length
        if len(prompt) < self.min_prompt_length:
            issues.append("Prompt too short")
        elif len(prompt) > self.max_prompt_length:
            issues.append("Prompt too long")
        
        # Check for placeholder remnants
        if '{' in prompt and '}' in prompt:
            issues.append("Unfilled template variables detected")
        
        # Check for instruction clarity
        if not any(word in prompt.lower() for word in ['write', 'generate', 'complete', 'respond']):
            issues.append("No clear instruction found")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'length': len(prompt),
            'estimated_tokens': len(prompt.split()) * 1.3  # Rough token estimate
        }
    
    def suggest_improvements(self, template: PromptTemplate) -> List[str]:
        """Suggest improvements for template quality."""
        suggestions = []
        validation = self.validate_template(template)
        
        if not validation['is_valid']:
            suggestions.extend([f"Fix: {issue}" for issue in validation['issues']])
        
        # Complexity-based suggestions
        if template.difficulty_level == DifficultyLevel.SIMPLE:
            suggestions.append("Consider adding more context for better training")
        elif template.difficulty_level == DifficultyLevel.ADVANCED:
            suggestions.append("Ensure sufficient examples for complex patterns")
        
        return suggestions
    
    def get_template_metrics(self, template: PromptTemplate) -> Dict[str, Any]:
        """Calculate comprehensive template metrics."""
        return {
            'length': len(template.template),
            'word_count': len(template.template.split()),
            'variable_count': template.template.count('{'),
            'complexity_score': template.get_complexity_score(),
            'content_types': [ct.value for ct in template.content_types],
            'difficulty_level': template.difficulty_level.value,
            'prompt_mode': template.prompt_mode.value
        }
    
    def generate_advanced_prompt(self, document: Dict[str, Any], chunk_content: str, 
                               template_name: Optional[str] = None,
                               difficulty_level: Optional[DifficultyLevel] = None,
                               conversation_context: Optional[str] = None) -> str:
        """Generate advanced prompt with speaker adaptation and difficulty control."""
        # Select template based on criteria
        if template_name is None:
            template_name = self._select_optimal_template(document, chunk_content, difficulty_level)
        
        template = self.templates.get(template_name)
        if not template:
            template = self.templates["simple_completion"]  # Default fallback
        
        # Get speaker profile for enhanced context
        author_name = document.get('author', 'Unknown Author')
        speaker_profile = self.speaker_manager.get_profile(author_name)
        
        # Prepare enhanced template variables
        template_vars = self._prepare_template_variables(
            document, chunk_content, speaker_profile, conversation_context
        )
        
        # Generate and validate prompt
        prompt = template.format(**template_vars)
        validation = self.prompt_validator.validate_generated_prompt(prompt)
        
        if not validation['is_valid']:
            # Try fallback template if validation fails
            fallback_template = self.templates["simple_completion"]
            prompt = fallback_template.format(**template_vars)
        
        return prompt
    
    def _select_template_for_content_type(self, content_type: ContentType) -> str:
        """Select best template for detected content type."""
        template_mapping = {
            ContentType.SERMON: "style_mimicry",
            ContentType.TESTIMONY: "testimony", 
            ContentType.PRAYER: "prayer",
            ContentType.TEACHING: "style_mimicry",
            ContentType.DOCTRINAL: "topical_response",
            ContentType.PERSONAL_STORY: "testimony",
            ContentType.GENERAL: "style_mimicry"
        }
        return template_mapping.get(content_type, "style_mimicry")
    
    def _extract_main_topic(self, content: str) -> str:
        """Extract main topic from content for topical templates."""
        # Simple topic extraction based on common conference themes
        topics = {
            'faith': ['faith', 'believe', 'trust', 'confidence'],
            'family': ['family', 'marriage', 'children', 'parent'],
            'service': ['serve', 'service', 'help', 'ministry'],
            'repentance': ['repent', 'forgive', 'mercy', 'change'],
            'temple': ['temple', 'covenant', 'ordinance', 'sacred'],
            'priesthood': ['priesthood', 'authority', 'keys', 'ordain'],
            'testimony': ['testimony', 'witness', 'know', 'spirit'],
            'revelation': ['revelation', 'inspiration', 'guidance', 'direction']
        }
        
        content_lower = content.lower()
        topic_scores = {}
        
        for topic, keywords in topics.items():
            score = sum(content_lower.count(keyword) for keyword in keywords)
            if score > 0:
                topic_scores[topic] = score
        
        return max(topic_scores, key=topic_scores.get) if topic_scores else "gospel principles"


class ContentClassifier:
    """Classifier for General Conference content types."""
    
    def __init__(self):
        """Initialize classifier with keyword patterns."""
        self.classification_patterns = {
            ContentType.SERMON: {
                'title_keywords': ['teach', 'doctrine', 'gospel', 'principle'],
                'content_keywords': ['brothers and sisters', 'my dear friends', 'let me share', 'i want to speak'],
                'structural_patterns': [r'first.*second.*third', r'let me.*suggest', r'i invite you']
            },
            ContentType.TESTIMONY: {
                'title_keywords': ['testimony', 'witness', 'know'],
                'content_keywords': ['i know', 'i testify', 'my testimony', 'bear witness', 'spirit whispers'],
                'structural_patterns': [r'i know that', r'my testimony.*is', r'i witness']
            },
            ContentType.PRAYER: {
                'title_keywords': ['prayer', 'pray', 'heavenly father'],
                'content_keywords': ['heavenly father', 'dear father', 'we thank thee', 'we pray', 'amen'],
                'structural_patterns': [r'our.*father', r'we.*grateful', r'in.*name.*jesus']
            },
            ContentType.TEACHING: {
                'title_keywords': ['learning', 'teach', 'education', 'instruction'],
                'content_keywords': ['learn', 'understand', 'study', 'ponder', 'scripture'],
                'structural_patterns': [r'we learn.*', r'scripture teaches', r'let us.*understand']
            },
            ContentType.DOCTRINAL: {
                'title_keywords': ['doctrine', 'principle', 'truth', 'revelation'],
                'content_keywords': ['doctrine', 'revealed', 'truth', 'principle', 'eternal'],
                'structural_patterns': [r'doctrine.*teaches', r'truth.*is', r'principle.*of']
            },
            ContentType.PERSONAL_STORY: {
                'title_keywords': ['experience', 'story', 'learned'],
                'content_keywords': ['when i was', 'i remember', 'experience taught', 'learned that'],
                'structural_patterns': [r'i.*remember', r'years ago', r'experience.*taught']
            }
        }
    
    def classify_content(self, title: str, content: str) -> ContentType:
        """Classify content type based on title and content analysis."""
        title_lower = title.lower()
        content_lower = content.lower()
        
        scores = {}
        
        for content_type, patterns in self.classification_patterns.items():
            score = 0
            
            # Title keyword matching (weighted higher)
            for keyword in patterns['title_keywords']:
                if keyword in title_lower:
                    score += 3
            
            # Content keyword matching
            for keyword in patterns['content_keywords']:
                score += content_lower.count(keyword)
            
            # Structural pattern matching
            for pattern in patterns['structural_patterns']:
                matches = len(re.findall(pattern, content_lower))
                score += matches * 2
            
            scores[content_type] = score
        
        # Return highest scoring type, default to GENERAL
        if not scores or max(scores.values()) == 0:
            return ContentType.GENERAL
        
        return max(scores, key=scores.get)


def create_training_prompt(document: Dict[str, Any], chunk_content: str, 
                          template_name: Optional[str] = None) -> str:
    """
    Create training prompt for document chunk.
    
    Args:
        document: Document metadata (author, title, date, etc.)
        chunk_content: The text content to create prompt for
        template_name: Optional specific template to use
    
    Returns:
        Formatted training prompt string
    """
    engine = EnhancedPromptEngine()
    return engine.generate_prompt(document, chunk_content, template_name)


def analyze_prompt_effectiveness(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze prompt template effectiveness across document corpus.
    
    Args:
        documents: List of documents to analyze
    
    Returns:
        Analysis report with template usage statistics and recommendations
    """
    classifier = ContentClassifier()
    engine = EnhancedPromptEngine()
    
    type_distribution = {}
    template_usage = {}
    
    for doc in documents:
        content = doc.get('content', '')
        content_type = classifier.classify_content(doc.get('title', ''), content)
        template_name = engine._select_template_for_content_type(content_type)
        
        # Track distribution
        type_distribution[content_type.value] = type_distribution.get(content_type.value, 0) + 1
        template_usage[template_name] = template_usage.get(template_name, 0) + 1
    
    return {
        'content_type_distribution': type_distribution,
        'template_usage': template_usage,
        'total_documents': len(documents),
        'classification_coverage': len(type_distribution),
        'template_diversity': len(template_usage)
    }