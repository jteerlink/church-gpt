"""
Configuration Module

Configuration settings and validation for content reformatting.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class EncodingConfig:
    """Encoding fix configuration."""
    fixes: Dict[str, str] = field(default_factory=lambda: {
        'â': '"',      # Left double quote
        'â': '"',      # Right double quote  
        'â': "'",      # Apostrophe
        'Ã©': 'é',     # e with acute
        'Ã': 'À',      # A with grave
        'â¦': '…',     # Ellipsis
        'â': '—',      # Em dash
        'Â': ' ',      # Non-breaking space issue
        'â': '–',      # En dash
        'Ã¡': 'á',     # a with acute
        'Ã³': 'ó',     # o with acute
        'Ã­': 'í',     # i with acute
        'Ãº': 'ú',     # u with acute
        'Ã¼': 'ü',     # u with diaeresis
        'Ã±': 'ñ',     # n with tilde
        'Ã§': 'ç',     # c with cedilla
    })


@dataclass  
class SkipPatterns:
    """Patterns for content to skip."""
    session_files: List[str] = field(default_factory=lambda: [
        "saturday-morning-session",
        "sunday-afternoon-session", 
        "priesthood-session",
        "relief-society-session",
        "young-women-session",
        "general-young-women-meeting"
    ])
    
    toc_indicators: List[str] = field(default_factory=lambda: [
        "Authenticating...",
        "Contents",
        "Session"
    ])


@dataclass
class MetadataPatterns:
    """Patterns for metadata extraction."""
    conference_date: List[str] = field(default_factory=lambda: [
        r'(April|October)\s+(\d{4})',
        r'(\d{4})\s+(April|October)'
    ])
    
    author_name: List[str] = field(default_factory=lambda: [
        r'By\s+(Elder|President|Bishop|Sister)\s+([^\n\r]+?)(?:\n|$)',
        r'^(Elder|President|Bishop|Sister)\s+([^\n\r]+?)(?:\n|Of the)'
    ])
    
    title_indicators: List[str] = field(default_factory=lambda: [
        "By ", "Elder ", "President ", "Bishop ", "Sister "
    ])


@dataclass
class QualityConfig:
    """Quality assurance configuration."""
    min_title_length: int = 3
    min_body_length: int = 100
    min_author_length: int = 5
    require_date: bool = False
    require_author_title: bool = False
    max_encoding_issues: int = 50
    
    
@dataclass  
class ReformatterConfig:
    """Complete reformatter configuration."""
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    skip_patterns: SkipPatterns = field(default_factory=SkipPatterns)
    metadata_patterns: MetadataPatterns = field(default_factory=MetadataPatterns)
    quality: QualityConfig = field(default_factory=QualityConfig)
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'ReformatterConfig':
        """Load configuration from YAML file."""
        if not config_path.exists():
            return cls()  # Use defaults
            
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        # Create config with loaded data
        config = cls()
        
        if 'encoding' in data:
            config.encoding.fixes.update(data['encoding'].get('fixes', {}))
            
        if 'skip_patterns' in data:
            patterns = data['skip_patterns']
            if 'session_files' in patterns:
                config.skip_patterns.session_files = patterns['session_files']
            if 'toc_indicators' in patterns:
                config.skip_patterns.toc_indicators = patterns['toc_indicators']
                
        if 'metadata_patterns' in data:
            patterns = data['metadata_patterns']
            if 'conference_date' in patterns:
                config.metadata_patterns.conference_date = patterns['conference_date']
            if 'author_name' in patterns:
                config.metadata_patterns.author_name = patterns['author_name']
            if 'title_indicators' in patterns:
                config.metadata_patterns.title_indicators = patterns['title_indicators']
                
        if 'quality' in data:
            quality = data['quality']
            for key, value in quality.items():
                if hasattr(config.quality, key):
                    setattr(config.quality, key, value)
                    
        return config
        
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            'encoding': {
                'fixes': self.encoding.fixes
            },
            'skip_patterns': {
                'session_files': self.skip_patterns.session_files,
                'toc_indicators': self.skip_patterns.toc_indicators
            },
            'metadata_patterns': {
                'conference_date': self.metadata_patterns.conference_date,
                'author_name': self.metadata_patterns.author_name,
                'title_indicators': self.metadata_patterns.title_indicators
            },
            'quality': {
                'min_title_length': self.quality.min_title_length,
                'min_body_length': self.quality.min_body_length,
                'min_author_length': self.quality.min_author_length,
                'require_date': self.quality.require_date,
                'require_author_title': self.quality.require_author_title,
                'max_encoding_issues': self.quality.max_encoding_issues
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)


def get_default_config() -> ReformatterConfig:
    """Get default reformatter configuration."""
    return ReformatterConfig()