"""
Metadata Extraction Module

Extracts and validates metadata from scraped content.
"""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime


class MetadataExtractor:
    """Extracts structured metadata from scraped content."""
    
    # Conference date patterns
    CONFERENCE_DATE_PATTERNS = [
        r'(April|October)\s+(\d{4})',
        r'(\d{4})\s+(April|October)',
        r'(April|October)\s*\d{4}',
    ]
    
    # Author patterns with various formats
    AUTHOR_PATTERNS = [
        r'By\s+(Elder|President|Bishop|Sister)\s+([^\n\r]+?)(?:\n|$)',
        r'^(Elder|President|Bishop|Sister)\s+([^\n\r]+?)(?:\n|Of the)',
        r'By\s+([^\n\r]+?)(?:\n|$)',
    ]
    
    # Author title patterns
    AUTHOR_TITLE_PATTERNS = [
        r'Of the (Quorum of the Twelve Apostles)',
        r'Of the (First Presidency)',
        r'Of the (Seventy)',
        r'Of the (Presiding Bishopric)', 
        r'Of the (Relief Society General Presidency)',
        r'Of the (Young Women General Presidency)',
        r'Of the (Primary General Presidency)',
        r'(Relief Society General President)',
        r'(Young Women General President)',
        r'(Primary General President)',
    ]
    
    # Month name mapping
    MONTH_MAPPING = {
        'January': '01', 'February': '02', 'March': '03', 'April': '04',
        'May': '05', 'June': '06', 'July': '07', 'August': '08',
        'September': '09', 'October': '10', 'November': '11', 'December': '12'
    }

    def __init__(self):
        self.current_year = datetime.now().year

    def parse_conference_date(self, content: str, filepath: Optional[Path] = None) -> Optional[str]:
        """Parse conference date from content or filepath."""
        # First try to extract from content
        for pattern in self.CONFERENCE_DATE_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    if groups[0].isdigit():
                        year, month = groups[0], groups[1]
                    else:
                        month, year = groups[0], groups[1]
                    return f"{month} {year}"
        
        # Try to extract from filepath if available
        if filepath:
            path_str = str(filepath)
            
            # Look for pattern like 1995-04 or 2008-10
            date_match = re.search(r'(\d{4})-(\d{2})', path_str)
            if date_match:
                year, month = date_match.groups()
                month_name = 'April' if month == '04' else 'October' if month == '10' else None
                if month_name:
                    return f"{month_name} {year}"
            
            # Look for year in path
            year_match = re.search(r'(\d{4})', path_str)
            if year_match:
                year = year_match.group(1)
                # Guess based on common conference months
                if 'april' in path_str.lower() or '04' in path_str:
                    return f"April {year}"
                elif 'october' in path_str.lower() or '10' in path_str:
                    return f"October {year}"
        
        return None

    def extract_speaker_info(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract speaker name and title."""
        lines = content.split('\n')
        
        author = None
        author_title = None
        
        # Look for author in first 20 lines
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            if not line:
                continue
                
            # Try different author patterns
            for pattern in self.AUTHOR_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    groups = match.groups()
                    if len(groups) == 2:
                        title, name = groups
                        author = f"{title} {name}".strip()
                    else:
                        author = groups[0].strip()
                    break
            
            if author:
                # Look for author title in next few lines
                for j in range(i, min(i + 5, len(lines))):
                    title_line = lines[j].strip()
                    for title_pattern in self.AUTHOR_TITLE_PATTERNS:
                        title_match = re.search(title_pattern, title_line)
                        if title_match:
                            author_title = title_match.group(1)
                            break
                    if author_title:
                        break
                break
        
        return author, author_title

    def detect_talk_title(self, content: str) -> Optional[str]:
        """Extract the talk title."""
        lines = content.split('\n')
        
        # Skip empty lines and look for first substantial line
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip obvious navigation/metadata
            if any(skip in line.lower() for skip in [
                'authenticating', 'contents', 'session', 'april', 'october'
            ]):
                continue
                
            # Skip author lines
            if any(auth in line for auth in ['By ', 'Elder ', 'President ', 'Bishop ', 'Sister ']):
                continue
                
            # Skip author title lines
            if any(title in line for title in ['Of the Quorum', 'Of the First', 'Of the Seventy']):
                continue
            
            # This should be the title
            if len(line) > 3 and not line.isdigit():
                return line
        
        return None

    def validate_metadata(self, metadata: Dict[str, Optional[str]]) -> bool:
        """Validate that essential metadata is present and reasonable."""
        # Must have title
        if not metadata.get('title') or len(metadata['title'].strip()) < 3:
            return False
        
        # Must have author for conference talks
        if not metadata.get('author'):
            return False
            
        # Author should contain appropriate title
        author = metadata['author']
        if not any(title in author for title in ['Elder', 'President', 'Bishop', 'Sister']):
            return False
        
        # Date validation
        date = metadata.get('date')
        if date:
            date_match = re.match(r'(April|October) (\d{4})', date)
            if date_match:
                year = int(date_match.group(2))
                if not (1970 <= year <= self.current_year + 1):
                    return False
        
        return True

    def extract_all_metadata(self, content: str, filepath: Optional[Path] = None) -> Dict[str, Optional[str]]:
        """Extract all metadata from content."""
        title = self.detect_talk_title(content)
        author, author_title = self.extract_speaker_info(content)
        date = self.parse_conference_date(content, filepath)
        
        metadata = {
            'title': title,
            'author': author,
            'author_title': author_title,
            'date': date
        }
        
        return metadata

    def infer_missing_metadata(self, metadata: Dict[str, Optional[str]], filepath: Optional[Path] = None) -> Dict[str, Optional[str]]:
        """Infer missing metadata from available information."""
        result = metadata.copy()
        
        # Infer date from filepath if missing
        if not result.get('date') and filepath:
            result['date'] = self.parse_conference_date('', filepath)
        
        # Clean up author name formatting
        if result.get('author'):
            author = result['author']
            # Remove redundant titles
            author = re.sub(r'\s+(Elder|President|Bishop|Sister)\s+', ' ', author)
            # Clean up whitespace
            author = ' '.join(author.split())
            result['author'] = author
        
        # Standardize author title
        if result.get('author_title'):
            title = result['author_title']
            # Standardize common variations
            title_mapping = {
                'Quorum of the Twelve Apostles': 'Of the Quorum of the Twelve Apostles',
                'First Presidency': 'Of the First Presidency',
                'Seventy': 'Of the Seventy',
                'Presiding Bishopric': 'Of the Presiding Bishopric'
            }
            for original, standardized in title_mapping.items():
                if original in title:
                    result['author_title'] = standardized
                    break
        
        return result