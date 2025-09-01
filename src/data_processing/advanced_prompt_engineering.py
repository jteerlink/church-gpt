"""
Advanced Prompt Engineering System for LDS General Conference Fine-Tuning
Phase 4: Implements sophisticated prompt templates with speaker adaptation and conversation modes.
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


class AdvancedPromptTemplate:
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
        # Handle missing keys gracefully
        safe_kwargs = {}
        for key, value in kwargs.items():
            if value is None:
                safe_kwargs[key] = f"[{key}]"
            elif isinstance(value, list):
                safe_kwargs[key] = ", ".join(str(v) for v in value)
            else:
                safe_kwargs[key] = str(value)
        
        try:
            return self.template.format(**safe_kwargs)
        except KeyError as e:
            # Extract the missing key and provide a placeholder
            missing_key = str(e).strip("'\"")
            safe_kwargs[missing_key] = f"[{missing_key}]"
            try:
                return self.template.format(**safe_kwargs)
            except:
                return self.template + f"\n\n[Missing template variable: {e}]"
    
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
    
    def create_speaker_context(self, speaker_name: str) -> Dict[str, str]:
        """Create context dictionary for speaker-specific templates."""
        profile = self.get_profile(speaker_name)
        if not profile:
            return {
                'speaking_style': 'authoritative and caring',
                'common_phrases': 'My dear brothers and sisters',
                'rhetorical_patterns': 'clear teaching',
                'topic_expertise': 'gospel principles'
            }
        
        return {
            'speaking_style': profile.speaking_style,
            'common_phrases': ', '.join(profile.common_phrases[:3]),  # Top 3
            'rhetorical_patterns': ', '.join(profile.rhetorical_patterns[:2]),  # Top 2
            'topic_expertise': ', '.join(profile.topic_expertise[:3])  # Top 3
        }


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
    
    def validate_template(self, template: AdvancedPromptTemplate) -> Dict[str, Any]:
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


class AdvancedPromptEngine:
    """Advanced prompt engineering system with speaker adaptation and difficulty progression."""
    
    def __init__(self):
        """Initialize prompt engine with advanced templates and speaker profiles."""
        self.templates = self._create_templates()
        self.speaker_manager = SpeakerProfileManager()
        self.prompt_validator = PromptValidator()
        self.content_classifier = self._create_content_classifier()
    
    def _create_templates(self) -> Dict[str, AdvancedPromptTemplate]:
        """Create advanced prompt templates with difficulty progression and conversation modes."""
        templates = {}
        
        # Simple completion template
        templates["simple_completion"] = AdvancedPromptTemplate(
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
        templates["conversation"] = AdvancedPromptTemplate(
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
        templates["speaker_adapted"] = AdvancedPromptTemplate(
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
        templates["interview"] = AdvancedPromptTemplate(
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
        templates["instruction_following"] = AdvancedPromptTemplate(
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
        templates["testimony"] = AdvancedPromptTemplate(
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
        
        # Complex testimony template
        templates["complex_testimony"] = AdvancedPromptTemplate(
            name="complex_testimony",
            template="""Provide a profound testimony about {topic} in {author}'s distinctive style, incorporating deep doctrinal understanding and personal conviction.

Advanced Context:
- Author: {author} with {speaking_style} delivery
- Rhetorical approach: {rhetorical_patterns}
- Signature elements: {common_phrases}
- Doctrinal expertise: {topic_expertise}

Craft a testimony that demonstrates both personal witness and doctrinal depth about {topic}:

""",
            description="Complex testimony with doctrinal depth",
            content_types=[ContentType.TESTIMONY, ContentType.DOCTRINAL, ContentType.GENERAL],
            difficulty_level=DifficultyLevel.COMPLEX,
            prompt_mode=PromptMode.STYLE_TRANSFER,
            requires_context=True
        )
        
        # Prayer template
        templates["prayer"] = AdvancedPromptTemplate(
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
    
    def _create_content_classifier(self):
        """Create simplified content classifier."""
        class SimpleClassifier:
            def classify_content(self, title: str, content: str) -> ContentType:
                title_lower = title.lower()
                content_lower = content.lower()
                
                if any(word in title_lower for word in ['testimony', 'witness']):
                    return ContentType.TESTIMONY
                elif any(word in title_lower for word in ['prayer', 'pray']):
                    return ContentType.PRAYER
                elif any(word in content_lower for word in ['i know', 'i testify']):
                    return ContentType.TESTIMONY
                elif any(word in content_lower for word in ['heavenly father', 'amen']):
                    return ContentType.PRAYER
                else:
                    return ContentType.GENERAL
        
        return SimpleClassifier()
    
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
        
        # If validation fails and it's not due to length issues, try to use the prompt anyway
        # if it's a context-required template (interview, conversation)
        if not validation['is_valid']:
            # Check if it's just unfilled variables in a context template
            if template.requires_context and any("unfilled" in issue.lower() for issue in validation['issues']):
                # For context templates, we may have intentional placeholders
                return prompt
            else:
                # Try fallback template if validation fails for other reasons
                fallback_template = self.templates["simple_completion"]
                prompt = fallback_template.format(**template_vars)
        
        return prompt
    
    def _select_optimal_template(self, document: Dict[str, Any], chunk_content: str, 
                               difficulty_level: Optional[DifficultyLevel]) -> str:
        """Select optimal template based on content, speaker, and difficulty."""
        # Classify content
        content_type = self.content_classifier.classify_content(
            document.get('title', ''), chunk_content
        )
        
        # Get speaker profile for difficulty preference
        author_name = document.get('author', 'Unknown Author')
        speaker_profile = self.speaker_manager.get_profile(author_name)
        
        # Use speaker's difficulty preference if not specified
        if difficulty_level is None and speaker_profile:
            difficulty_level = speaker_profile.difficulty_preference
        elif difficulty_level is None:
            difficulty_level = DifficultyLevel.MODERATE
        
        # Select template based on content type and difficulty
        template_candidates = []
        for template_name, template in self.templates.items():
            if (content_type in template.content_types or ContentType.GENERAL in template.content_types) and \
               template.difficulty_level == difficulty_level:
                template_candidates.append(template_name)
        
        if template_candidates:
            return template_candidates[0]  # Use first match for predictability in tests
        
        # If no exact difficulty match, try to find templates that match content type
        content_candidates = []
        for template_name, template in self.templates.items():
            if content_type in template.content_types or ContentType.GENERAL in template.content_types:
                content_candidates.append((template_name, template.difficulty_level))
        
        if content_candidates:
            # Sort by difficulty level distance and return closest
            target_complexity = difficulty_level.value if hasattr(difficulty_level, 'value') else 'moderate'
            complexity_order = ['simple', 'moderate', 'complex', 'advanced']
            target_index = complexity_order.index(target_complexity) if target_complexity in complexity_order else 1
            
            content_candidates.sort(key=lambda x: abs(complexity_order.index(x[1].value) - target_index))
            return content_candidates[0][0]
        
        # Fallback selection based on content type only
        fallback_mapping = {
            ContentType.TESTIMONY: "testimony",
            ContentType.PRAYER: "prayer",
            ContentType.CONVERSATION: "conversation",
            ContentType.INTERVIEW: "interview"
        }
        
        return fallback_mapping.get(content_type, "simple_completion")
    
    def _prepare_template_variables(self, document: Dict[str, Any], chunk_content: str,
                                   speaker_profile: Optional[SpeakerProfile],
                                   conversation_context: Optional[str]) -> Dict[str, Any]:
        """Prepare comprehensive template variables with speaker adaptation."""
        author_name = document.get('author', 'Unknown Author')
        
        # Base variables
        template_vars = {
            'author': author_name,
            'title': document.get('title', 'Untitled'),
            'date': document.get('date', 'Unknown Date'),
            'partial_text': chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
            'topic': self._extract_main_topic(chunk_content)
        }
        
        # Add speaker-specific context
        if speaker_profile:
            speaker_context = self.speaker_manager.create_speaker_context(author_name)
            template_vars.update(speaker_context)
        else:
            # Default speaker characteristics
            template_vars.update({
                'speaking_style': 'authoritative and caring',
                'common_phrases': 'My dear brothers and sisters',
                'rhetorical_patterns': 'clear teaching',
                'topic_expertise': 'gospel principles'
            })
        
        # Add conversation context if provided
        if conversation_context:
            template_vars['question'] = conversation_context
        else:
            template_vars['question'] = f"Can you share your thoughts about {template_vars['topic']}?"
        
        return template_vars
    
    def _extract_main_topic(self, content: str) -> str:
        """Extract main topic from content for topical templates."""
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
    
    def get_difficulty_progression(self, current_level: DifficultyLevel) -> List[str]:
        """Get template names for difficulty progression."""
        level_templates = {}
        for template_name, template in self.templates.items():
            level = template.difficulty_level
            if level not in level_templates:
                level_templates[level] = []
            level_templates[level].append(template_name)
        
        return level_templates.get(current_level, [])
    
    def create_conversation_prompt(self, author: str, question: str, 
                                 topic_context: Optional[str] = None) -> str:
        """Create a conversation-style prompt for specific question."""
        document = {
            'author': author,
            'title': f"Conversation about {topic_context or 'gospel topics'}",
            'date': 'Recent'
        }
        
        return self.generate_advanced_prompt(
            document=document,
            chunk_content=topic_context or question,
            template_name="conversation",
            conversation_context=question
        )
    
    def validate_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """Validate all templates and return results."""
        validation_results = {}
        
        for template_name, template in self.templates.items():
            validation_results[template_name] = self.prompt_validator.validate_template(template)
        
        return validation_results


# Factory function for creating training prompts
def create_advanced_training_prompt(document: Dict[str, Any], chunk_content: str, 
                                  template_name: Optional[str] = None,
                                  difficulty_level: Optional[DifficultyLevel] = None) -> str:
    """
    Create advanced training prompt for document chunk.
    
    Args:
        document: Document metadata (author, title, date, etc.)
        chunk_content: The text content to create prompt for
        template_name: Optional specific template to use
        difficulty_level: Optional difficulty level for training progression
    
    Returns:
        Formatted training prompt string
    """
    engine = AdvancedPromptEngine()
    return engine.generate_advanced_prompt(document, chunk_content, template_name, difficulty_level)


def analyze_prompt_effectiveness(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze prompt template effectiveness across document corpus.
    
    Args:
        documents: List of documents to analyze
    
    Returns:
        Analysis report with template usage statistics and recommendations
    """
    engine = AdvancedPromptEngine()
    
    type_distribution = {}
    template_usage = {}
    difficulty_distribution = {}
    speaker_coverage = {}
    
    for doc in documents:
        content = doc.get('content', '')
        author = doc.get('author', 'Unknown')
        
        # Classify content and select template
        content_type = engine.content_classifier.classify_content(doc.get('title', ''), content)
        template_name = engine._select_optimal_template(doc, content, None)
        template = engine.templates.get(template_name)
        
        # Track distributions
        type_distribution[content_type.value] = type_distribution.get(content_type.value, 0) + 1
        template_usage[template_name] = template_usage.get(template_name, 0) + 1
        
        if template:
            difficulty = template.difficulty_level.value
            difficulty_distribution[difficulty] = difficulty_distribution.get(difficulty, 0) + 1
        
        speaker_coverage[author] = speaker_coverage.get(author, 0) + 1
    
    return {
        'content_type_distribution': type_distribution,
        'template_usage': template_usage,
        'difficulty_distribution': difficulty_distribution,
        'speaker_coverage': speaker_coverage,
        'total_documents': len(documents),
        'classification_coverage': len(type_distribution),
        'template_diversity': len(template_usage),
        'supported_speakers': len([s for s in speaker_coverage.keys() 
                                 if engine.speaker_manager.get_profile(s) is not None])
    }