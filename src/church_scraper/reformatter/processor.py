"""
Text Processing Module

Handles text cleaning, encoding fixes, and content extraction.
"""

import re
from typing import List, Tuple, Optional
from .analyzer import EncodingIssue


class TextProcessor:
    """Processes and cleans scraped text content."""
    
    # Encoding fixes mapping
    ENCODING_FIXES = {
        'â': '"',      # Left double quote
        'â': '"',      # Right double quote  
        'â': "'",      # Apostrophe
        'Ã©': 'é',     # e with acute
        'Ã': 'À',      # A with grave
        'â¦': '…',     # Ellipsis
        'â': '—',      # Em dash
        'Â': ' ',      # Non-breaking space issue (HeberÂ J. -> Heber J.)
        'â': '–',      # En dash
        'Ã¡': 'á',     # a with acute
        'Ã³': 'ó',     # o with acute
        'Ã­': 'í',     # i with acute
        'Ãº': 'ú',     # u with acute
        'Ã¼': 'ü',     # u with diaeresis
        'Ã±': 'ñ',     # n with tilde
        'Ã§': 'ç',     # c with cedilla
        
        # Additional corrupted encoding patterns (Windows-1252 artifacts)
        '\x80\x9c': '"',     # Left double quote
        '\x80\x9d': '"',     # Right double quote
        '\x80\x94': '—',     # Em dash
        '\x80\x99': "'",     # Apostrophe/right single quote
        '\x80\x93': '–',     # En dash
        '\x80\x98': "'",     # Left single quote
        '\x80\xa6': '…',     # Ellipsis
        '\x80\xa0': ' ',     # Non-breaking space
    }
    
    # Table of contents patterns to remove
    TOC_REMOVAL_PATTERNS = [
        # Authentication and year range headers
        r"Authenticating\.\.\.\s*",
        r"\d{4}â\d{4}\s+",
        
        # 2023+ Format: Complete TOC block removal
        # Pattern: "General Conference [Month] [Year] general conference[massive_toc_content][Title][Month] [Year] general conference"
        r"General Conference\s+\w+\s+\d{4}\s+general conference.*?\w+\s*October \d{4} general conference",
        
        # Contents and session navigation (legacy formats)
        r"Contents\s*",
        r"Saturday Morning Session.*?Sunday Afternoon Session",
        r"(?:Saturday|Sunday) (?:Morning|Afternoon) Session.*?(?=\n\n|\n[A-Z])",
        
        # Session listing patterns
        r"(?:Priesthood|Relief Society|Young Women) Session.*?(?=\n\n|\n[A-Z])",
        
        # Talk listing in TOC (before actual content starts)
        r"^.*?Session\s*.*?(?=\n\n[A-Z][a-z].*?\n)",
        
        # Remove lines that are just navigation elements
        r"^\s*(?:The |Elder |President |Bishop |Sister ).*?(?:Quorum|Presidency|Seventy|Bishopric)\s*$",
        
        # Time stamps and session markers at start
        r"^\s*\d{1,2}:\d{2}.*$",
    ]
    
    # Patterns to identify content start
    CONTENT_START_PATTERNS = [
        r"^[A-Z][a-zA-Z\s:,-]+\n(?:By |Elder |President |Bishop |Sister )",
        r"^[A-Z][a-zA-Z\s:,-]+\n[A-Z][a-z].*?\n",
    ]
    
    # Notes section patterns
    NOTES_PATTERNS = [
        r"\nNotes?\s*\n",
        r"\nReferences?\s*\n",
        r"\nFootnotes?\s*\n",
    ]

    def fix_encoding(self, text: str) -> str:
        """Fix encoding issues in text."""
        result = text
        
        for old, new in self.ENCODING_FIXES.items():
            result = result.replace(old, new)
        
        return result

    def clean_table_of_contents(self, text: str) -> str:
        """Remove table of contents and navigation contamination."""
        result = text
        
        # Check if content was already cleaned by analyzer (has timestamp+title pattern)
        lines = result.split('\n')
        if lines and re.match(r'^\d{1,2}:\d{2}[A-Za-z]', lines[0].strip()):
            # Content already cleaned by analyzer, just apply pattern cleaning
            for pattern in self.TOC_REMOVAL_PATTERNS:
                result = re.sub(pattern, '', result, flags=re.MULTILINE | re.DOTALL)
            return result.strip()
        
        # Apply TOC removal patterns
        for pattern in self.TOC_REMOVAL_PATTERNS:
            result = re.sub(pattern, '', result, flags=re.MULTILINE | re.DOTALL)
        
        # Check if this looks like already-cleaned content (no TOC contamination indicators)
        cleaned_lines = result.split('\n')
        has_toc_indicators = any(
            any(indicator in line.lower() for indicator in ['contents', 'session', 'authenticating'])
            for line in cleaned_lines[:10]  # Check first 10 lines
        )
        
        if not has_toc_indicators:
            # Content appears clean, return as-is
            return result.strip()
        
        # Find where actual content starts (legacy logic for uncleaned content)
        content_start = 0
        
        # Look for the actual talk title and author
        for i, line in enumerate(cleaned_lines):
            line = line.strip()
            if not line:
                continue
                
            # Skip obvious TOC/navigation lines
            if any(skip in line.lower() for skip in ['contents', 'session', 'authenticating']):
                continue
                
            # Look for title followed by author pattern
            if (i + 1 < len(cleaned_lines) and 
                len(line) > 10 and 
                not line.endswith(':')):
                
                next_line = cleaned_lines[i + 1].strip()
                # More precise author pattern matching - must be at start of line or after "By "
                if (next_line.startswith(('By Elder ', 'By President ', 'By Bishop ', 'By Sister ')) or
                    re.match(r'^(Elder|President|Bishop|Sister)\s+[A-Z]', next_line)):
                    content_start = i
                    break
                
            # Or standalone author line indicating content start
            if (line.startswith(('By Elder ', 'By President ', 'By Bishop ', 'By Sister ')) or
                re.match(r'^(Elder|President|Bishop|Sister)\s+[A-Z]', line)):
                # Title should be in previous line
                if i > 0 and cleaned_lines[i-1].strip():
                    content_start = i - 1
                else:
                    content_start = i
                break
        
        if content_start > 0:
            result = '\n'.join(cleaned_lines[content_start:])
        
        return result.strip()

    def extract_body_content(self, text: str) -> str:
        """Extract main body content, excluding metadata and notes."""
        lines = text.split('\n')
        
        # Find content boundaries
        body_start = 0
        body_end = len(lines)
        
        # Skip metadata at the beginning (but only check first few lines)
        found_body_start = False
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Only skip metadata in the first 5 lines to avoid skipping actual content
            if i < 5:
                # Skip title, author, date lines
                if (any(marker in line for marker in ['By ', 'Elder ', 'President ', 'Bishop ', 'Sister ']) or
                    any(title in line for title in ['Of the Quorum', 'Of the First', 'Of the Seventy', 'Of the Presiding']) or
                    re.match(r'(April|October) \d{4}', line)):
                    continue
            
            # This should be start of actual content
            body_start = i
            found_body_start = True
            break
        
        # If we didn't find body start, start from beginning
        if not found_body_start:
            body_start = 0
        
        # Find notes section
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if re.match(r'^Notes?$', line, re.IGNORECASE):
                body_end = i
                break
        
        body_lines = lines[body_start:body_end]
        return '\n'.join(body_lines).strip()

    def extract_notes_section(self, text: str) -> Optional[str]:
        """Extract notes/references section if present."""
        # Find notes section
        for pattern in self.NOTES_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract everything after the notes header
                notes_start = match.end()
                notes_content = text[notes_start:].strip()
                if notes_content:
                    return f"Notes\n{notes_content}"
        
        return None

    def clean_timestamp_prefix(self, title: str) -> str:
        """Remove timestamp prefixes like '14:56', '12:23', '4:28' from titles."""
        if not title:
            return title
        cleaned = re.sub(r'^\d{1,2}:\d{2}', '', title).strip()
        return cleaned if cleaned else title

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks."""
        # Remove excessive blank lines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        
        # Remove leading/trailing empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        return '\n'.join(lines)

    def clean_content_artifacts(self, text: str) -> str:
        """Remove content artifacts and formatting issues."""
        result = text
        
        # Remove session time stamps that leak into content (but preserve timestamp+title)
        # Only remove standalone timestamps or session markers, not title lines
        result = re.sub(r'^\d{1,2}:\d{2}\s*$', '', result, flags=re.MULTILINE)  # Standalone timestamps only
        result = re.sub(r'^\d{1,2}:\d{2}\s+(Session|Morning|Afternoon|Evening)\b.*$', '', result, flags=re.MULTILINE)  # Session timestamps
        
        # Remove duplicate titles that appear in content
        lines = result.split('\n')
        if len(lines) > 2:
            title = lines[0].strip()
            # Remove duplicates of the title from content body
            for i in range(2, min(10, len(lines))):
                if lines[i].strip() == title:
                    lines[i] = ''
        
        result = '\n'.join(lines)
        
        # Remove excessive punctuation artifacts
        result = re.sub(r'[â]{2,}', '"', result)
        result = re.sub(r'\.{4,}', '...', result)
        
        return result

    def process_text(self, text: str) -> str:
        """Complete text processing pipeline."""
        # Step 1: Fix encoding issues
        result = self.fix_encoding(text)
        
        # Step 2: Clean table of contents contamination
        result = self.clean_table_of_contents(result)
        
        # Step 3: Clean content artifacts
        result = self.clean_content_artifacts(result)
        
        # Step 4: Normalize whitespace
        result = self.normalize_whitespace(result)
        
        return result

    def split_content_and_notes(self, text: str) -> Tuple[str, Optional[str]]:
        """Split text into main content and notes section."""
        processed_text = self.process_text(text)
        
        # Extract notes
        notes = self.extract_notes_section(processed_text)
        
        # Get body content (everything except notes)
        if notes:
            # Remove notes section from main content
            notes_start = processed_text.lower().rfind('notes')
            if notes_start > 0:
                body_content = processed_text[:notes_start].strip()
            else:
                body_content = self.extract_body_content(processed_text)
        else:
            body_content = self.extract_body_content(processed_text)
        
        return body_content, notes