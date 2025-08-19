"""
Content Analysis Module

Analyzes scraped content to determine type, structure, and identify issues.
"""

import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
from enum import Enum


class ContentType(Enum):
    """Content type classification."""
    CONFERENCE_TALK = "conference_talk"
    LIAHONA_ARTICLE = "liahona_article"
    MAGAZINE_PROFILE = "magazine_profile"
    SESSION_HEADER = "session_header"
    CONTENTS_PAGE = "contents_page"
    TABLE_OF_CONTENTS = "table_of_contents"
    CITATIONS_ONLY = "citations_only"
    SHORT_REMARKS = "short_remarks"
    INVALID = "invalid"


class EncodingIssue(NamedTuple):
    """Represents an encoding issue found in content."""
    position: int
    original: str
    suggested_fix: str
    confidence: float


class StructureReport(NamedTuple):
    """Report on content structure analysis."""
    has_title: bool
    has_author: bool
    has_date: bool
    has_body: bool
    has_notes: bool
    toc_contamination: bool
    duplicate_title: bool
    line_count: int
    issues: List[str]


class ContentAnalyzer:
    """Analyzes scraped content structure and identifies issues."""
    
    # Session patterns that indicate non-talk content
    SESSION_PATTERNS = [
        r"saturday-morning-session",
        r"sunday-afternoon-session", 
        r"priesthood-session",
        r"relief-society-session",
        r"young-women-session",
        r"general-young-women-meeting"
    ]
    
    # Table of contents indicators
    TOC_PATTERNS = [
        r"Authenticating\.\.\.",
        r"\d{4}[â—–-]\d{4}",  # Year ranges like 1990â1999, 1990-1999
        r"ContentsSaturday Morning Session",
        r"Saturday Morning SessionThe",
        r"Sunday Afternoon SessionThe",
        r"Contents(Saturday|Sunday|Priesthood)",
        r"^Contents\s*$",  # Magazine contents pages
        r"ContentsContents",  # Duplicate contents markers
        r"\d{4}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}Contents",
        
        # 2023+ Format: Massive concatenated TOC detection
        r"General Conference\s+\w+\s+\d{4}\s+general conference",  # TOC start marker
        r"Contents.*?Session.*?Session",  # Multiple sessions in contents
        r"\w+\s*October \d{4} general conference",  # TOC end marker
    ]
    
    # Encoding issue patterns
    ENCODING_PATTERNS = [
        (r'â', '"', 0.95),  # Left quote
        (r'â', '"', 0.95),  # Right quote
        (r'â', "'", 0.95),  # Apostrophe
        (r'Ã©', 'é', 0.90),  # e with acute
        (r'Ã', 'À', 0.85),   # A with grave
        (r'â¦', '…', 0.90),  # Ellipsis
        (r'â', '—', 0.90),   # Em dash
        (r'Â', ' ', 0.85),   # Non-breaking space issue
        (r'â', '–', 0.85),   # En dash
        (r'Ã¡', 'á', 0.90),  # a with acute
        (r'Ã³', 'ó', 0.90),  # o with acute
    ]
    
    # Metadata patterns
    AUTHOR_PATTERNS = [
        r"By (Elder|President|Bishop|Sister) ([^\n]+)",
        r"(Elder|President|Bishop|Sister) ([^\n]+)",
    ]
    
    TITLE_PATTERNS = [
        r"Of the (Quorum of the Twelve Apostles|First Presidency|Seventy)",
        r"Of the (Presiding Bishopric|Relief Society|Young Women)",
    ]
    
    DATE_PATTERNS = [
        r"(April|October) (\d{4})",
        r"(\d{4})\s+(April|October)",
    ]

    def detect_content_type(self, content: str, filepath: Optional[Path] = None) -> ContentType:
        """Detect the type of content based on patterns and filename."""
        if filepath:
            filename = filepath.name.lower()
            
            # Check for session files
            for pattern in self.SESSION_PATTERNS:
                if re.search(pattern, filename):
                    return ContentType.SESSION_HEADER
            
            # Check for contents files
            if 'contents' in filename:
                return ContentType.CONTENTS_PAGE
        
        lines = content.split('\n')
        content_text = '\n'.join(lines[:20])  # First 20 lines for analysis
        
        # Check for table of contents contamination (but exclude simple timestamp cases)
        has_timestamp_only = bool(lines and re.match(r'^\d{1,2}:\d{2}[A-Za-z]', lines[0].strip()))
        
        # Check if this is a PURE TOC file vs a conference talk WITH TOC contamination
        if not has_timestamp_only:  # Only check TOC patterns if it's not just a timestamp
            toc_matches = 0
            for pattern in self.TOC_PATTERNS:
                if re.search(pattern, content_text):
                    toc_matches += 1
            
            # Only classify as pure TOC if:
            # 1. Multiple TOC patterns match (indicating heavy TOC content), AND
            # 2. No clear conference talk structure (author, substantial content)
            if toc_matches >= 2:
                has_author = any(re.search(pattern, line) for pattern in self.AUTHOR_PATTERNS for line in lines[:30])
                has_substantial_content = len([line for line in lines if len(line.strip()) > 50]) > 3
                
                # If no author and little content, likely pure TOC
                if not has_author and not has_substantial_content:
                    return ContentType.TABLE_OF_CONTENTS
        
        # Check if title starts with "Contents"
        if lines and lines[0].strip().lower() == 'contents':
            return ContentType.CONTENTS_PAGE
        
        # Check for timestamp contamination in title
        title_has_timestamp = bool(lines and re.match(r'^\d{1,2}:\d{2}', lines[0].strip()))
        
        # Check if it has proper talk structure
        has_author = any(re.search(pattern, line) for pattern in self.AUTHOR_PATTERNS for line in lines[:30])
        has_title = len(lines) > 0 and len(lines[0].strip()) > 5
        has_substantial_content = len([line for line in lines if len(line.strip()) > 50])
        
        # Check for citations-only content
        if has_title and has_author and has_substantial_content < 3:
            # Check if most content appears to be citations
            citation_lines = sum(1 for line in lines if re.search(r'^\d+\.|^See |^".*?"|^\w+\s+\d+:', line.strip()))
            if citation_lines > has_substantial_content:
                return ContentType.CITATIONS_ONLY
        
        # Determine content type based on source and characteristics
        if filepath and 'liahona' in str(filepath):
            if has_author and has_title and has_substantial_content > 10:
                # Check if it's a profile/feature article
                if re.search(r'Making Friends|By \w+ \w+\s*Church Magazines', content):
                    return ContentType.MAGAZINE_PROFILE
                return ContentType.LIAHONA_ARTICLE
            elif has_substantial_content < 5:
                return ContentType.SHORT_REMARKS
        elif filepath and 'general-conference' in str(filepath):
            if has_author and has_title:
                if has_substantial_content < 5:
                    return ContentType.SHORT_REMARKS
                elif has_substantial_content > 3:
                    return ContentType.CONFERENCE_TALK
        
        # If no filepath but has good structure, infer type
        if has_author and has_title and has_substantial_content > 3:
            return ContentType.CONFERENCE_TALK
        elif has_title and has_substantial_content > 1:
            return ContentType.SHORT_REMARKS
        
        return ContentType.INVALID

    def extract_metadata(self, content: str) -> Dict[str, Optional[str]]:
        """Extract metadata from content."""
        lines = content.split('\n')
        metadata = {
            'title': None,
            'author': None,
            'author_title': None,
            'date': None
        }
        
        # Extract title (usually first non-empty line)
        for line in lines:
            if line.strip() and not re.search(r'Authenticating|Contents|Session', line):
                metadata['title'] = line.strip()
                break
        
        # Extract author information
        for line in lines[:30]:  # Check first 30 lines
            for pattern in self.AUTHOR_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    if len(match.groups()) == 2:
                        title, name = match.groups()
                        metadata['author'] = f"{title} {name}"
                    else:
                        metadata['author'] = match.group(0)
                    break
        
        # Extract author title
        for line in lines[:30]:
            for pattern in self.TITLE_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    metadata['author_title'] = match.group(0)
                    break
        
        # Extract date
        for line in lines[:30]:
            for pattern in self.DATE_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    if len(match.groups()) == 2:
                        month, year = match.groups()
                        if month in ['April', 'October']:
                            metadata['date'] = f"{month} {year}"
                        else:
                            metadata['date'] = f"{month} {year}"
                    break
        
        return metadata

    def identify_encoding_issues(self, content: str) -> List[EncodingIssue]:
        """Identify encoding issues in content."""
        issues = []
        
        for pattern, fix, confidence in self.ENCODING_PATTERNS:
            for match in re.finditer(pattern, content):
                issues.append(EncodingIssue(
                    position=match.start(),
                    original=match.group(0),
                    suggested_fix=fix,
                    confidence=confidence
                ))
        
        return sorted(issues, key=lambda x: x.position)

    def validate_structure(self, content: str) -> StructureReport:
        """Validate content structure and identify issues."""
        lines = content.split('\n')
        issues = []
        
        # Check for basic components
        has_title = bool(lines and lines[0].strip())
        has_author = any(re.search(pattern, line) for pattern in self.AUTHOR_PATTERNS for line in lines[:30])
        has_date = any(re.search(pattern, line) for pattern in self.DATE_PATTERNS for line in lines[:30])
        has_body = len([line for line in lines if len(line.strip()) > 50]) > 3
        has_notes = any('Notes' in line or 'notes' in line for line in lines[-20:])
        
        # Check for contamination
        toc_contamination = any(re.search(pattern, content) for pattern in self.TOC_PATTERNS)
        if toc_contamination:
            issues.append("Table of contents contamination detected")
        
        # Check for duplicate titles
        if has_title:
            title = lines[0].strip()
            duplicate_title = content.count(title) > 1
            if duplicate_title:
                issues.append("Duplicate title found")
        else:
            duplicate_title = False
            
        # Structure validation
        if not has_title:
            issues.append("Missing title")
        if not has_author:
            issues.append("Missing author information")
        if not has_date:
            issues.append("Missing date information")
        if not has_body:
            issues.append("Insufficient body content")
            
        # Encoding issues
        encoding_issues = self.identify_encoding_issues(content)
        if encoding_issues:
            issues.append(f"Found {len(encoding_issues)} encoding issues")
        
        return StructureReport(
            has_title=has_title,
            has_author=has_author,
            has_date=has_date,
            has_body=has_body,
            has_notes=has_notes,
            toc_contamination=toc_contamination,
            duplicate_title=duplicate_title,
            line_count=len(lines),
            issues=issues
        )

    def extract_clean_content(self, content: str) -> str:
        """Extract content after TOC contamination and timestamp issues."""
        lines = content.split('\n')
        
        # Handle simple timestamp prefix case first
        if lines and re.match(r'^\d{1,2}:\d{2}[A-Za-z]', lines[0].strip()):
            # This is just a timestamp prefix, clean it and return
            lines[0] = self.clean_timestamp_prefix(lines[0])
            return '\n'.join(lines)
        
        # Check if content starts clean (no TOC contamination)
        if not any(re.search(pattern, content[:1000]) for pattern in self.TOC_PATTERNS):
            return content
        
        # Special handling for 2023+ format with massive TOC block
        # Look for the complete TOC block pattern: "General Conference...massive_content...TitleOctober Year general conference"
        toc_block_pattern = r'General Conference\s+\w+\s+\d{4}\s+general conference.*?\w+\s*October \d{4} general conference'
        toc_block_match = re.search(toc_block_pattern, content, flags=re.DOTALL)
        if toc_block_match:
            # Remove the entire TOC block and get the remaining content
            toc_end_pos = toc_block_match.end()
            remaining_content = content[toc_end_pos:].strip()
            
            # Look for the first substantial line after the TOC block
            remaining_lines = remaining_content.split('\n')
            for i, line in enumerate(remaining_lines):
                line = line.strip()
                if line and len(line) > 5:  # Skip empty or very short lines
                    # Found the start of actual content
                    return '\n'.join(remaining_lines[i:])
        
        # Fallback: Find actual content start after TOC contamination (legacy logic)
        title = lines[0].strip() if lines else ""
        
        # Look for timestamp+title pattern (e.g., "4:28Welcome to Conference")
        for i, line in enumerate(lines):
            timestamp_match = re.match(r'^\d{1,2}:\d{2}(.+)', line.strip())
            if timestamp_match:
                # Found timestamp+title, this is likely the real start
                clean_title = timestamp_match.group(1)
                lines[i] = clean_title
                return '\n'.join(lines[i:])
        
        # Look for repeated title indicating content start
        if title and len(title) > 10:
            clean_title = re.sub(r'^\d{1,2}:\d{2}', '', title).strip()
            for i in range(10, min(100, len(lines))):
                line = lines[i].strip()
                # Check for exact match or timestamp+title match
                if (line == title or line == clean_title or 
                    (clean_title in line and len(line) < len(clean_title) + 10)):
                    return '\n'.join(lines[i:])
        
        # If we can't find a clean start point, return original
        return content

    def clean_timestamp_prefix(self, title: str) -> str:
        """Remove timestamp prefixes like '14:56', '12:23', '4:28' from titles."""
        if not title:
            return title
        return re.sub(r'^\d{1,2}:\d{2}', '', title).strip()