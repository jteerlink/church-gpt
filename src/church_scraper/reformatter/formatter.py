"""
Content Formatting Module

Formats processed content according to standardized templates.
"""

import re
from typing import Dict, Optional
from .analyzer import ContentType


class ContentFormatter:
    """Formats content according to standardized template."""
    
    # Content type-specific minimum content thresholds
    CONTENT_THRESHOLDS = {
        ContentType.CONFERENCE_TALK: 500,     # Normal talks (reduced from 1000)
        ContentType.SHORT_REMARKS: 100,       # Presidential remarks (reduced from 200)
        ContentType.LIAHONA_ARTICLE: 300,     # Magazine articles (reduced from 500)
        ContentType.MAGAZINE_PROFILE: 200,    # Feature profiles (reduced from 300)
        ContentType.CITATIONS_ONLY: 50,       # Citations/references only
        ContentType.SESSION_HEADER: 50,       # Session headers (reduced from 100)
        ContentType.CONTENTS_PAGE: 50,        # Contents pages (reduced from 100)
        ContentType.TABLE_OF_CONTENTS: 50,    # TOC pages (reduced from 100)
        ContentType.INVALID: 50,              # Invalid content (reduced from 100)
    }
    
    # Standard template for formatted content
    STANDARD_TEMPLATE = """DATE: {date}
TITLE: {title}
AUTHOR: {author}
AUTHOR TITLE: {author_title}
BODY:
{body}{notes_section}"""
    
    # Alternative template when some metadata is missing
    MINIMAL_TEMPLATE = """TITLE: {title}
AUTHOR: {author}
BODY:
{body}{notes_section}"""

    def format_to_standard(self, 
                          title: str,
                          author: str,
                          body: str,
                          date: Optional[str] = None,
                          author_title: Optional[str] = None,
                          notes: Optional[str] = None) -> str:
        """Format content to standard template."""
        
        # Prepare notes section
        notes_section = ""
        if notes:
            # Ensure notes section starts with newline
            if not body.endswith('\n'):
                notes_section = f"\n{notes}"
            else:
                notes_section = notes
        
        # Choose template based on available metadata
        if date and author_title:
            template = self.STANDARD_TEMPLATE
            return template.format(
                date=date,
                title=title,
                author=author,
                author_title=author_title,
                body=body,
                notes_section=notes_section
            )
        else:
            template = self.MINIMAL_TEMPLATE
            formatted = template.format(
                title=title,
                author=author,
                body=body,
                notes_section=notes_section
            )
            
            # Add date if available
            if date:
                formatted = f"DATE: {date}\n{formatted}"
            
            # Add author title if available
            if author_title:
                lines = formatted.split('\n')
                author_line_idx = next(i for i, line in enumerate(lines) if line.startswith('AUTHOR: '))
                lines.insert(author_line_idx + 1, f"AUTHOR TITLE: {author_title}")
                formatted = '\n'.join(lines)
            
            return formatted

    def apply_template(self, metadata: Dict[str, Optional[str]], body: str, notes: Optional[str] = None) -> str:
        """Apply template using metadata dictionary."""
        return self.format_to_standard(
            title=metadata.get('title', 'Unknown Title'),
            author=metadata.get('author', 'Unknown Author'),
            body=body,
            date=metadata.get('date'),
            author_title=metadata.get('author_title'),
            notes=notes
        )

    def validate_output(self, formatted_content: str, content_type: ContentType = ContentType.CONFERENCE_TALK) -> bool:
        """Validate formatted output meets standards for the specific content type."""
        lines = formatted_content.split('\n')
        
        # Must have minimum required fields
        required_patterns = [
            r'^TITLE: .+',
            r'^AUTHOR: .+',
            r'^BODY:',
        ]
        
        content_text = '\n'.join(lines)
        
        for pattern in required_patterns:
            if not re.search(pattern, content_text, re.MULTILINE):
                return False
        
        # Validate structure
        if 'BODY:' not in content_text:
            return False
        
        body_start = content_text.find('BODY:')
        body_content = content_text[body_start + 5:].strip()
        
        # Check content length based on content type
        min_length = self.CONTENT_THRESHOLDS.get(content_type, 100)
        if len(body_content) < min_length:
            return False
        
        return True

    def clean_formatting(self, content: str) -> str:
        """Clean up formatting issues in final output."""
        # Remove excessive blank lines
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Ensure proper spacing after metadata fields
        content = re.sub(r'^(DATE|TITLE|AUTHOR|AUTHOR TITLE): (.+)$', r'\1: \2', content, flags=re.MULTILINE)
        
        # Ensure BODY: section has proper formatting
        content = re.sub(r'^BODY:\s*\n\s*', 'BODY:\n', content, flags=re.MULTILINE)
        
        # Clean up notes section formatting
        content = re.sub(r'\nNotes\s*\n\s*\n', '\nNotes\n', content)
        
        return content.strip()

    def format_notes_section(self, notes_content: str) -> str:
        """Format notes section consistently."""
        if not notes_content:
            return ""
        
        # Ensure it starts with "Notes" header
        if not notes_content.strip().startswith('Notes'):
            notes_content = f"Notes\n{notes_content}"
        
        # Clean up formatting
        lines = notes_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove excessive indentation
            line = line.strip()
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1]:  # Keep single blank lines
                cleaned_lines.append('')
        
        return '\n'.join(cleaned_lines)

    def standardize_quotes(self, content: str) -> str:
        """Standardize quotation marks in content."""
        # Replace smart quotes with standard quotes
        content = content.replace('"', '"')
        content = content.replace('"', '"')
        content = content.replace(''', "'")
        content = content.replace(''', "'")
        
        return content

    def format_scripture_references(self, content: str) -> str:
        """Standardize scripture reference formatting."""
        # Common scripture reference patterns
        patterns = [
            # Handle references like "1 Nephi 8:27"
            (r'(\d+)\s+([A-Za-z]+)\s+(\d+):(\d+)', r'\1 \2 \3:\4'),
            # Handle "D&C 123:12"
            (r'(D&C)\s+(\d+):(\d+)', r'\1 \2:\3'),
            # Handle "3 Ne. 18:21"
            (r'(\d+)\s+([A-Za-z]+)\.\s+(\d+):(\d+)', r'\1 \2. \3:\4'),
        ]
        
        result = content
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        
        return result

    def complete_formatting(self, metadata: Dict[str, Optional[str]], body: str, notes: Optional[str] = None) -> str:
        """Complete formatting pipeline."""
        # Format notes if present
        formatted_notes = None
        if notes:
            formatted_notes = self.format_notes_section(notes)
        
        # Apply main template
        formatted = self.apply_template(metadata, body, formatted_notes)
        
        # Clean up formatting
        formatted = self.clean_formatting(formatted)
        
        # Standardize quotes and references
        formatted = self.standardize_quotes(formatted)
        formatted = self.format_scripture_references(formatted)
        
        return formatted