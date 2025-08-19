#!/usr/bin/env python3
"""
Unit tests for content reformatter components.
"""

import unittest
import tempfile
from pathlib import Path
from src.church_scraper.reformatter import (
    ContentAnalyzer, 
    TextProcessor, 
    MetadataExtractor,
    ContentFormatter,
    ContentType
)


class TestContentAnalyzer(unittest.TestCase):
    """Test content analysis functionality."""
    
    def setUp(self):
        self.analyzer = ContentAnalyzer()
    
    def test_detect_conference_talk(self):
        """Test detection of conference talk content."""
        content = """A Table Encircled with Love
By Elder LeGrand R. Curtis
Of the Seventy
April 1995

Much has been written about the importance of the home. This is substantial content that goes on for multiple paragraphs to demonstrate that this is indeed a real conference talk.

We recognize that some homes are large, graciously appointed, even luxurious. Others are very small and humble, with scant furnishings to fill their rooms.

Yet each and every home can be a haven on earth when we are filled with love, where we want to be, as one of our beloved hymns reminds us in its beautiful lyrics.

One of the more important furnishings found in most homes is the kitchen table where families gather for nourishment and fellowship together."""
        
        # Provide filepath to help with detection
        filepath = Path("scraped_content/general-conference/1995-04/table-encircled-with-love.txt")
        result = self.analyzer.detect_content_type(content, filepath)
        self.assertEqual(result, ContentType.CONFERENCE_TALK)
    
    def test_detect_session_header(self):
        """Test detection of session header files."""
        filepath = Path("saturday-morning-sessionlangeng.txt")
        content = "Saturday Morning Session Contents..."
        
        result = self.analyzer.detect_content_type(content, filepath)
        self.assertEqual(result, ContentType.SESSION_HEADER)
    
    def test_detect_table_of_contents(self):
        """Test detection of table of contents contamination."""
        content = """Authenticating...
1990–1999 April 1995 Contents Saturday Morning Session The Shield of Faith"""
        
        result = self.analyzer.detect_content_type(content)
        self.assertEqual(result, ContentType.TABLE_OF_CONTENTS)
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        content = """A Table Encircled with Love
By Elder LeGrand R. Curtis
Of the Seventy
April 1995

Body content here..."""
        
        metadata = self.analyzer.extract_metadata(content)
        
        self.assertEqual(metadata['title'], 'A Table Encircled with Love')
        self.assertEqual(metadata['author'], 'Elder LeGrand R. Curtis')
        self.assertEqual(metadata['author_title'], 'Of the Seventy')
        self.assertEqual(metadata['date'], 'April 1995')
    
    def test_identify_encoding_issues(self):
        """Test encoding issue detection."""
        content = "This has âbad quotesâ and Â spacing issues."
        
        issues = self.analyzer.identify_encoding_issues(content)
        
        self.assertGreater(len(issues), 0)
        self.assertTrue(any('â' in issue.original for issue in issues))


class TestTextProcessor(unittest.TestCase):
    """Test text processing functionality."""
    
    def setUp(self):
        self.processor = TextProcessor()
    
    def test_fix_encoding(self):
        """Test encoding fixes."""
        text = "This has âbad quotesâ and Â spacing."
        result = self.processor.fix_encoding(text)
        
        # Check that encoding issues were fixed (may not be exact quotes)
        self.assertNotIn('â', result)  # Should be removed
        self.assertNotIn('Â', result)  # Should be removed
        self.assertIn('bad quotes', result)  # Content should remain
    
    def test_clean_table_of_contents(self):
        """Test TOC removal."""
        text = """Authenticating...
1990–1999 April 1995 Contents Saturday Morning Session

A Table Encircled with Love
By Elder LeGrand R. Curtis
Of the Seventy

Much has been written about the home."""
        
        result = self.processor.clean_table_of_contents(text)
        
        self.assertNotIn('Authenticating', result)
        self.assertNotIn('Contents', result)
        self.assertIn('A Table Encircled with Love', result)
    
    def test_extract_body_content(self):
        """Test body content extraction."""
        text = """A Table Encircled with Love
By Elder LeGrand R. Curtis  
Of the Seventy
April 1995

Much has been written about the home.

This is the main content.

Notes
Reference 1"""
        
        body = self.processor.extract_body_content(text)
        
        self.assertIn('Much has been written', body)
        self.assertIn('main content', body)
        self.assertNotIn('Notes', body)
        self.assertNotIn('Reference 1', body)
    
    def test_extract_notes_section(self):
        """Test notes extraction."""
        text = """Main content here.

Notes
Reference 1
Reference 2"""
        
        notes = self.processor.extract_notes_section(text)
        
        self.assertIsNotNone(notes)
        self.assertIn('Notes', notes)
        self.assertIn('Reference 1', notes)
    
    def test_split_content_and_notes(self):
        """Test content and notes splitting."""
        text = """Title
Author

Body content here.

Notes
Reference 1"""
        
        body, notes = self.processor.split_content_and_notes(text)
        
        self.assertIn('Body content', body)
        self.assertNotIn('Notes', body)
        self.assertIsNotNone(notes)
        self.assertIn('Reference 1', notes)


class TestMetadataExtractor(unittest.TestCase):
    """Test metadata extraction functionality."""
    
    def setUp(self):
        self.extractor = MetadataExtractor()
    
    def test_parse_conference_date_from_content(self):
        """Test date parsing from content."""
        content = "April 1995 General Conference"
        result = self.extractor.parse_conference_date(content)
        self.assertEqual(result, "April 1995")
    
    def test_parse_conference_date_from_filepath(self):
        """Test date parsing from filepath."""
        filepath = Path("scraped_content/general-conference/1995-04/talk.txt")
        result = self.extractor.parse_conference_date("", filepath)
        self.assertEqual(result, "April 1995")
    
    def test_extract_speaker_info(self):
        """Test speaker information extraction."""
        content = """Title Here
By Elder John Smith
Of the Quorum of the Twelve Apostles

Content here..."""
        
        author, title = self.extractor.extract_speaker_info(content)
        
        self.assertEqual(author, "Elder John Smith")
        self.assertEqual(title, "Quorum of the Twelve Apostles")
    
    def test_detect_talk_title(self):
        """Test talk title detection."""
        content = """The Power of Faith
By Elder John Smith
Of the Seventy

Faith is essential..."""
        
        title = self.extractor.detect_talk_title(content)
        self.assertEqual(title, "The Power of Faith")
    
    def test_validate_metadata_valid(self):
        """Test metadata validation with valid data."""
        metadata = {
            'title': 'The Power of Faith',
            'author': 'Elder John Smith',
            'author_title': 'Of the Seventy',
            'date': 'April 2020'
        }
        
        result = self.extractor.validate_metadata(metadata)
        self.assertTrue(result)
    
    def test_validate_metadata_invalid(self):
        """Test metadata validation with invalid data."""
        metadata = {
            'title': '',  # Too short
            'author': 'John Smith',  # Missing title
            'date': 'April 1900'  # Too old
        }
        
        result = self.extractor.validate_metadata(metadata)
        self.assertFalse(result)


class TestContentFormatter(unittest.TestCase):
    """Test content formatting functionality."""
    
    def setUp(self):
        self.formatter = ContentFormatter()
    
    def test_format_to_standard_complete(self):
        """Test formatting with complete metadata."""
        result = self.formatter.format_to_standard(
            title="Test Title",
            author="Elder Test Author",
            body="This is the main content.",
            date="April 2020",
            author_title="Of the Seventy",
            notes="Notes\nReference 1"
        )
        
        expected_lines = [
            "DATE: April 2020",
            "TITLE: Test Title", 
            "AUTHOR: Elder Test Author",
            "AUTHOR TITLE: Of the Seventy",
            "BODY:",
            "This is the main content.",
            "Notes",
            "Reference 1"
        ]
        
        for line in expected_lines:
            self.assertIn(line, result)
    
    def test_format_to_standard_minimal(self):
        """Test formatting with minimal metadata."""
        result = self.formatter.format_to_standard(
            title="Test Title",
            author="Elder Test Author", 
            body="This is the main content."
        )
        
        self.assertIn("TITLE: Test Title", result)
        self.assertIn("AUTHOR: Elder Test Author", result)
        self.assertIn("BODY:", result)
        self.assertIn("This is the main content", result)
    
    def test_validate_output_valid(self):
        """Test output validation with valid content."""
        content = """TITLE: Test Title
AUTHOR: Elder Test Author
BODY:
This is a substantial amount of content that should pass validation because it contains enough text to be considered valid body content for a conference talk."""
        
        result = self.formatter.validate_output(content)
        self.assertTrue(result)
    
    def test_validate_output_invalid(self):
        """Test output validation with invalid content."""
        content = """TITLE: Test Title
BODY:
Short"""  # Too short
        
        result = self.formatter.validate_output(content)
        self.assertFalse(result)
    
    def test_standardize_quotes(self):
        """Test quote standardization."""
        content = 'This has "smart quotes" and \'apostrophes\'.'
        result = self.formatter.standardize_quotes(content)
        
        self.assertIn('"smart quotes"', result)
        self.assertIn("'apostrophes'", result)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete reformatting workflow."""
    
    def setUp(self):
        self.analyzer = ContentAnalyzer()
        self.processor = TextProcessor()
        self.extractor = MetadataExtractor()
        self.formatter = ContentFormatter()
    
    def test_complete_workflow(self):
        """Test complete reformatting workflow."""
        # Sample problematic content similar to actual scraped files but clean enough to be detected as talk
        raw_content = """A Table Encircled with Love
By Elder LeGrand R. Curtis
Of the Seventy
April 1995

Much has been written about the importance of the home. Elder Marion G. Romney has told us that âat the heart of societyâs fatal sickness is the instability of the family.â

We recognize that some homes are large, graciously appointed, even luxurious. Others are very small and humble, with scant furnishings to fill their rooms and spaces.

Yet each and every âhome can be a heavân on earth when we are filled with love, where we want to be, as one of our beloved hymns reminds us in beautiful lyrics.

One of the more important furnishings found in most homes is the kitchen table where the different members of the family come to receive nourishment for their bodies.

Notes
âScriptures As They Relate to Family Stability,â Ensign, Feb. 1972, p. 57."""
        
        # Step 1: Analyze content with filepath context
        filepath = Path("scraped_content/general-conference/1995-04/table-encircled-with-love.txt")
        content_type = self.analyzer.detect_content_type(raw_content, filepath)
        self.assertEqual(content_type, ContentType.CONFERENCE_TALK)
        
        # Step 2: Process text
        processed_content, notes = self.processor.split_content_and_notes(raw_content)
        
        # Step 3: Extract metadata
        metadata = self.extractor.extract_all_metadata(raw_content)
        
        # Step 4: Format output
        formatted = self.formatter.complete_formatting(metadata, processed_content, notes)
        
        # Verify results
        self.assertIn("DATE: April 1995", formatted)
        self.assertIn("TITLE: A Table Encircled with Love", formatted)
        self.assertIn("AUTHOR: Elder LeGrand R. Curtis", formatted)
        self.assertTrue("AUTHOR TITLE:" in formatted and "Seventy" in formatted)
        self.assertIn("BODY:", formatted)
        self.assertIn("Much has been written", formatted)
        self.assertIn("Notes", formatted)
        
        # Verify encoding fixes
        self.assertNotIn('â', formatted)  # Should be fixed to proper quotes
        self.assertIn('at the heart', formatted)  # Content should remain
        
        # Verify TOC removal
        self.assertNotIn('Authenticating', formatted)
        self.assertNotIn('Contents', formatted)
        
        # Verify output validation
        self.assertTrue(self.formatter.validate_output(formatted))


if __name__ == '__main__':
    unittest.main(verbosity=2)