#!/usr/bin/env python3
"""
Unit tests for ConferenceScraper class.

Tests the General Conference scraping functionality with mocked HTTP responses.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

import requests
from requests.exceptions import RequestException

from src.church_scraper import ConferenceScraper, ScraperConfig


class TestConferenceScraper(unittest.TestCase):
    """Test cases for ConferenceScraper class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test output
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = ScraperConfig(
            start_year=1995,
            end_year=1997,
            content_type="conference",
            output_dir=self.temp_dir,
            delay=0.1,  # Faster for testing
            user_agent="Test-Scraper/1.0"
        )
        
        # Create scraper instance
        self.scraper = ConferenceScraper(self.config)
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_conference_urls(self):
        """Test generation of conference URLs for April and October sessions."""
        urls = self.scraper.get_conference_urls()
        
        # Should have 2 conferences per year (April and October) for 3 years (1995-1997)
        expected_count = 3 * 2  # 6 conferences total
        self.assertEqual(len(urls), expected_count)
        
        # Check first few URLs
        expected_urls = [
            (1995, 4, "https://www.churchofjesuschrist.org/study/general-conference/1995/04?lang=eng"),
            (1995, 10, "https://www.churchofjesuschrist.org/study/general-conference/1995/10?lang=eng"),
            (1996, 4, "https://www.churchofjesuschrist.org/study/general-conference/1996/04?lang=eng"),
            (1996, 10, "https://www.churchofjesuschrist.org/study/general-conference/1996/10?lang=eng"),
            (1997, 4, "https://www.churchofjesuschrist.org/study/general-conference/1997/04?lang=eng"),
            (1997, 10, "https://www.churchofjesuschrist.org/study/general-conference/1997/10?lang=eng"),
        ]
        
        self.assertEqual(urls, expected_urls)
    
    def test_get_conference_urls_single_year(self):
        """Test conference URL generation for a single year."""
        # Create config for single year
        single_year_config = ScraperConfig(
            start_year=2000,
            end_year=2000,
            output_dir=self.temp_dir
        )
        scraper = ConferenceScraper(single_year_config)
        
        urls = scraper.get_conference_urls()
        
        # Should have exactly 2 URLs for 2000
        self.assertEqual(len(urls), 2)
        self.assertEqual(urls[0], (2000, 4, "https://www.churchofjesuschrist.org/study/general-conference/2000/04?lang=eng"))
        self.assertEqual(urls[1], (2000, 10, "https://www.churchofjesuschrist.org/study/general-conference/2000/10?lang=eng"))
    
    @patch('src.church_scraper.core.ContentScraper.robust_get')
    def test_scrape_conference_page_success(self, mock_get):
        """Test successful scraping of conference page to extract talk URLs."""
        # Mock HTML response with talk links
        mock_html = """
        <html>
        <body>
            <div class="content">
                <a href="/study/general-conference/1995/04/talk-1">Talk 1</a>
                <a href="/study/general-conference/1995/04/talk-2">Talk 2</a>
                <a href="/study/general-conference/1995/04/talk-3">Talk 3</a>
                <a href="/study/general-conference/1995/04">Conference Index</a>
                <a href="/study/other-content/article">Other Content</a>
                <a href="https://external-site.com/link">External Link</a>
            </div>
        </body>
        </html>
        """
        
        # Configure mock response
        mock_response = Mock()
        mock_response.text = mock_html
        mock_get.return_value = mock_response
        
        # Test scraping
        conf_url = "https://www.churchofjesuschrist.org/study/general-conference/1995/04?lang=eng"
        talk_urls = self.scraper.scrape_conference_page(conf_url)
        
        # Should find 3 talk URLs (excluding conference index and other links)
        expected_urls = [
            "https://www.churchofjesuschrist.org/study/general-conference/1995/04/talk-1",
            "https://www.churchofjesuschrist.org/study/general-conference/1995/04/talk-2",
            "https://www.churchofjesuschrist.org/study/general-conference/1995/04/talk-3"
        ]
        
        self.assertEqual(len(talk_urls), 3)
        for expected_url in expected_urls:
            self.assertIn(expected_url, talk_urls)
        
        # Verify robust_get was called with correct URL
        mock_get.assert_called_once_with(conf_url)
    
    @patch('src.church_scraper.core.ContentScraper.robust_get')
    def test_scrape_conference_page_no_talks(self, mock_get):
        """Test scraping conference page with no talk links."""
        # Mock HTML response without talk links
        mock_html = """
        <html>
        <body>
            <div class="content">
                <p>No talks found on this page.</p>
                <a href="/study/other-content">Other Content</a>
            </div>
        </body>
        </html>
        """
        
        mock_response = Mock()
        mock_response.text = mock_html
        mock_get.return_value = mock_response
        
        conf_url = "https://www.churchofjesuschrist.org/study/general-conference/1995/04?lang=eng"
        talk_urls = self.scraper.scrape_conference_page(conf_url)
        
        # Should return empty list
        self.assertEqual(talk_urls, [])
    
    @patch('src.church_scraper.core.ContentScraper.robust_get')
    def test_scrape_conference_page_http_error(self, mock_get):
        """Test handling of HTTP errors when scraping conference page."""
        # Configure mock to raise RequestException
        mock_get.side_effect = RequestException("Network error")
        
        conf_url = "https://www.churchofjesuschrist.org/study/general-conference/1995/04?lang=eng"
        
        # Should raise RequestException
        with self.assertRaises(RequestException):
            self.scraper.scrape_conference_page(conf_url)
    
    @patch('src.church_scraper.core.ContentScraper.robust_get')
    def test_scrape_talk_success(self, mock_get):
        """Test successful scraping of individual talk content."""
        # Mock HTML response with talk content
        mock_html = """
        <html>
        <body>
            <div class="talk-content">
                <h1>Talk Title</h1>
                <p>This is the first paragraph of the talk.</p>
                <p>This is the second paragraph with more content.</p>
                <script>console.log('should be removed');</script>
                <style>.hidden { display: none; }</style>
            </div>
        </body>
        </html>
        """
        
        mock_response = Mock()
        mock_response.text = mock_html
        mock_get.return_value = mock_response
        
        talk_url = "https://www.churchofjesuschrist.org/study/general-conference/1995/04/talk-1"
        content = self.scraper.scrape_talk(talk_url)
        
        # Should extract clean text content
        self.assertIn("Talk Title", content)
        self.assertIn("This is the first paragraph", content)
        self.assertIn("This is the second paragraph", content)
        
        # Should not contain script or style content
        self.assertNotIn("console.log", content)
        self.assertNotIn("display: none", content)
        
        # Verify robust_get was called
        mock_get.assert_called_once_with(talk_url)
    
    @patch('src.church_scraper.core.ContentScraper.robust_get')
    def test_scrape_talk_empty_content(self, mock_get):
        """Test handling of talk page with no extractable content."""
        # Mock HTML response with no meaningful content
        mock_html = """
        <html>
        <body>
            <script>window.location = '/redirect';</script>
            <style>body { margin: 0; }</style>
        </body>
        </html>
        """
        
        mock_response = Mock()
        mock_response.text = mock_html
        mock_get.return_value = mock_response
        
        talk_url = "https://www.churchofjesuschrist.org/study/general-conference/1995/04/empty-talk"
        content = self.scraper.scrape_talk(talk_url)
        
        # Should return empty string for content with no text
        self.assertEqual(content.strip(), "")
    
    @patch('src.church_scraper.core.ContentScraper.robust_get')
    def test_scrape_talk_http_error(self, mock_get):
        """Test handling of HTTP errors when scraping talk."""
        # Configure mock to raise RequestException
        mock_get.side_effect = RequestException("Talk not found")
        
        talk_url = "https://www.churchofjesuschrist.org/study/general-conference/1995/04/missing-talk"
        
        # Should raise RequestException
        with self.assertRaises(RequestException):
            self.scraper.scrape_talk(talk_url)
    
    @patch('src.church_scraper.core.ConferenceScraper.scrape_talk')
    @patch('src.church_scraper.core.ConferenceScraper.scrape_conference_page')
    @patch('src.church_scraper.core.ConferenceScraper.get_conference_urls')
    def test_run_complete_workflow(self, mock_get_urls, mock_scrape_page, mock_scrape_talk):
        """Test complete scraping workflow with mocked methods."""
        # Mock conference URLs
        mock_get_urls.return_value = [
            (1995, 4, "https://www.churchofjesuschrist.org/study/general-conference/1995/04?lang=eng"),
            (1995, 10, "https://www.churchofjesuschrist.org/study/general-conference/1995/10?lang=eng")
        ]
        
        # Mock talk URLs from conference pages
        mock_scrape_page.side_effect = [
            [
                "https://www.churchofjesuschrist.org/study/general-conference/1995/04/talk-1",
                "https://www.churchofjesuschrist.org/study/general-conference/1995/04/talk-2"
            ],
            [
                "https://www.churchofjesuschrist.org/study/general-conference/1995/10/talk-3"
            ]
        ]
        
        # Mock talk content
        mock_scrape_talk.side_effect = [
            "Content of talk 1",
            "Content of talk 2", 
            "Content of talk 3"
        ]
        
        # Run the scraper
        self.scraper.run()
        
        # Verify all methods were called
        mock_get_urls.assert_called_once()
        self.assertEqual(mock_scrape_page.call_count, 2)
        self.assertEqual(mock_scrape_talk.call_count, 3)
        
        # Verify files were created
        april_dir = Path(self.temp_dir) / "general-conference" / "1995-04"
        october_dir = Path(self.temp_dir) / "general-conference" / "1995-10"
        
        self.assertTrue(april_dir.exists())
        self.assertTrue(october_dir.exists())
        
        # Check that talk files were created
        talk1_file = april_dir / "talk-1.txt"
        talk2_file = april_dir / "talk-2.txt"
        talk3_file = october_dir / "talk-3.txt"
        
        self.assertTrue(talk1_file.exists())
        self.assertTrue(talk2_file.exists())
        self.assertTrue(talk3_file.exists())
        
        # Verify file contents
        with open(talk1_file, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), "Content of talk 1")
        
        with open(talk2_file, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), "Content of talk 2")
        
        with open(talk3_file, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), "Content of talk 3")
    
    @patch('src.church_scraper.core.ConferenceScraper.scrape_talk')
    @patch('src.church_scraper.core.ConferenceScraper.scrape_conference_page')
    @patch('src.church_scraper.core.ConferenceScraper.get_conference_urls')
    def test_run_skip_existing_files(self, mock_get_urls, mock_scrape_page, mock_scrape_talk):
        """Test that existing files are skipped during scraping."""
        # Create existing file
        april_dir = Path(self.temp_dir) / "general-conference" / "1995-04"
        april_dir.mkdir(parents=True, exist_ok=True)
        existing_file = april_dir / "talk-1.txt"
        existing_file.write_text("Existing content", encoding='utf-8')
        
        # Mock conference URLs
        mock_get_urls.return_value = [
            (1995, 4, "https://www.churchofjesuschrist.org/study/general-conference/1995/04?lang=eng")
        ]
        
        # Mock talk URLs
        mock_scrape_page.return_value = [
            "https://www.churchofjesuschrist.org/study/general-conference/1995/04/talk-1",  # Existing
            "https://www.churchofjesuschrist.org/study/general-conference/1995/04/talk-2"   # New
        ]
        
        # Mock talk content
        mock_scrape_talk.return_value = "New talk content"
        
        # Run the scraper
        self.scraper.run()
        
        # Should only scrape the new talk (talk-2)
        mock_scrape_talk.assert_called_once_with(
            "https://www.churchofjesuschrist.org/study/general-conference/1995/04/talk-2"
        )
        
        # Existing file should remain unchanged
        with open(existing_file, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), "Existing content")
        
        # New file should be created
        new_file = april_dir / "talk-2.txt"
        self.assertTrue(new_file.exists())
        with open(new_file, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), "New talk content")
    
    @patch('src.church_scraper.core.ConferenceScraper.scrape_conference_page')
    @patch('src.church_scraper.core.ConferenceScraper.get_conference_urls')
    def test_run_handle_conference_page_error(self, mock_get_urls, mock_scrape_page):
        """Test handling of errors when scraping conference pages."""
        # Mock conference URLs
        mock_get_urls.return_value = [
            (1995, 4, "https://www.churchofjesuschrist.org/study/general-conference/1995/04?lang=eng"),
            (1995, 10, "https://www.churchofjesuschrist.org/study/general-conference/1995/10?lang=eng")
        ]
        
        # First conference page fails, second succeeds
        mock_scrape_page.side_effect = [
            RequestException("Conference page error"),
            ["https://www.churchofjesuschrist.org/study/general-conference/1995/10/talk-1"]
        ]
        
        # Should not raise exception, but continue with next conference
        try:
            self.scraper.run()
        except Exception as e:
            self.fail(f"run() raised an exception when it should have handled the error: {e}")
        
        # Both conference pages should have been attempted
        self.assertEqual(mock_scrape_page.call_count, 2)


if __name__ == '__main__':
    unittest.main()