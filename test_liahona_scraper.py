#!/usr/bin/env python3
"""
Unit tests for LiahonaScraper class.

Tests the Liahona magazine scraping functionality with mocked HTTP responses.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

import requests
from requests.exceptions import RequestException

from church_scraper import LiahonaScraper, ScraperConfig


class TestLiahonaScraper(unittest.TestCase):
    """Test cases for LiahonaScraper class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test output
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = ScraperConfig(
            start_year=2008,
            end_year=2009,
            content_type="liahona",
            output_dir=self.temp_dir,
            delay=0.1,  # Faster for testing
            user_agent="Test-Scraper/1.0"
        )
        
        # Create scraper instance
        self.scraper = LiahonaScraper(self.config)
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test LiahonaScraper initialization."""
        self.assertIsInstance(self.scraper.config, ScraperConfig)
        self.assertEqual(self.scraper.base_url, "https://www.churchofjesuschrist.org")
        self.assertIsNotNone(self.scraper.file_manager)
        self.assertIsNotNone(self.scraper.session)
    
    def test_get_monthly_urls(self):
        """Test generation of monthly Liahona URLs excluding April and October."""
        urls = self.scraper.get_monthly_urls()
        
        # Should have 10 months per year (12 - 2 conference months) * 2 years
        expected_count = 10 * 2
        self.assertEqual(len(urls), expected_count)
        
        # Check that April (4) and October (10) are excluded
        months_found = set()
        for year, month, url in urls:
            months_found.add(month)
            self.assertNotIn(month, [4, 10], "April and October should be excluded")
            
            # Verify URL format
            expected_url = f"https://www.churchofjesuschrist.org/study/liahona/{year}/{month:02d}?lang=eng"
            self.assertEqual(url, expected_url)
        
        # Should have months 1,2,3,5,6,7,8,9,11,12
        expected_months = {1, 2, 3, 5, 6, 7, 8, 9, 11, 12}
        self.assertEqual(months_found, expected_months)
    
    def test_get_monthly_urls_single_year(self):
        """Test monthly URL generation for a single year."""
        # Create config for single year
        single_year_config = ScraperConfig(
            start_year=2010,
            end_year=2010,
            content_type="liahona",
            output_dir=self.temp_dir
        )
        scraper = LiahonaScraper(single_year_config)
        
        urls = scraper.get_monthly_urls()
        
        # Should have 10 months for one year
        self.assertEqual(len(urls), 10)
        
        # All URLs should be for 2010
        for year, month, url in urls:
            self.assertEqual(year, 2010)
            self.assertNotIn(month, [4, 10])
    
    @patch('church_scraper.ContentScraper.robust_get')
    def test_scrape_monthly_page_success(self, mock_get):
        """Test successful scraping of monthly Liahona page."""
        # Mock HTML content with article links
        mock_html = """
        <html>
            <body>
                <div class="content">
                    <a href="/study/liahona/2008/01/article-1">Article 1</a>
                    <a href="/study/liahona/2008/01/article-2">Article 2</a>
                    <a href="/study/liahona/2008/01/article-3">Article 3</a>
                    <a href="/study/general-conference/2008/04/talk-1">Conference Talk</a>
                    <a href="/other/page">Other Page</a>
                </div>
            </body>
        </html>
        """
        
        # Mock response
        mock_response = Mock()
        mock_response.text = mock_html
        mock_get.return_value = mock_response
        
        # Test scraping
        month_url = "https://www.churchofjesuschrist.org/study/liahona/2008/01?lang=eng"
        article_urls = self.scraper.scrape_monthly_page(month_url)
        
        # Should find 3 article URLs (excluding conference talk and other page)
        expected_urls = [
            "https://www.churchofjesuschrist.org/study/liahona/2008/01/article-1",
            "https://www.churchofjesuschrist.org/study/liahona/2008/01/article-2",
            "https://www.churchofjesuschrist.org/study/liahona/2008/01/article-3"
        ]
        self.assertEqual(len(article_urls), 3)
        self.assertEqual(set(article_urls), set(expected_urls))
        
        # Verify robust_get was called with correct URL
        mock_get.assert_called_once_with(month_url)
    
    @patch('church_scraper.ContentScraper.robust_get')
    def test_scrape_monthly_page_no_articles(self, mock_get):
        """Test scraping monthly page with no article links."""
        # Mock HTML content without article links
        mock_html = """
        <html>
            <body>
                <div class="content">
                    <a href="/other/page">Other Page</a>
                    <a href="/study/general-conference/2008/04/talk-1">Conference Talk</a>
                </div>
            </body>
        </html>
        """
        
        # Mock response
        mock_response = Mock()
        mock_response.text = mock_html
        mock_get.return_value = mock_response
        
        # Test scraping
        month_url = "https://www.churchofjesuschrist.org/study/liahona/2008/01?lang=eng"
        article_urls = self.scraper.scrape_monthly_page(month_url)
        
        # Should find no article URLs
        self.assertEqual(len(article_urls), 0)
    
    @patch('church_scraper.ContentScraper.robust_get')
    def test_scrape_monthly_page_http_error(self, mock_get):
        """Test scraping monthly page with HTTP error."""
        # Mock HTTP error
        mock_get.side_effect = RequestException("HTTP 404 Not Found")
        
        # Test scraping should raise exception
        month_url = "https://www.churchofjesuschrist.org/study/liahona/2008/01?lang=eng"
        with self.assertRaises(RequestException):
            self.scraper.scrape_monthly_page(month_url)
    
    @patch('church_scraper.ContentScraper.robust_get')
    def test_scrape_article_success(self, mock_get):
        """Test successful scraping of individual article."""
        # Mock HTML content for article
        mock_html = """
        <html>
            <body>
                <div class="article-content">
                    <h1>Article Title</h1>
                    <p>This is the first paragraph of the article.</p>
                    <p>This is the second paragraph with more content.</p>
                </div>
            </body>
        </html>
        """
        
        # Mock response
        mock_response = Mock()
        mock_response.text = mock_html
        mock_get.return_value = mock_response
        
        # Test scraping
        article_url = "https://www.churchofjesuschrist.org/study/liahona/2008/01/test-article"
        content = self.scraper.scrape_article(article_url)
        
        # Should extract text content
        self.assertIn("Article Title", content)
        self.assertIn("first paragraph", content)
        self.assertIn("second paragraph", content)
        self.assertGreater(len(content), 0)
        
        # Verify robust_get was called with correct URL
        mock_get.assert_called_once_with(article_url)
    
    @patch('church_scraper.ContentScraper.robust_get')
    def test_scrape_article_empty_content(self, mock_get):
        """Test scraping article with empty or no content."""
        # Mock HTML content without meaningful text
        mock_html = """
        <html>
            <body>
                <script>var x = 1;</script>
                <style>.class { color: red; }</style>
            </body>
        </html>
        """
        
        # Mock response
        mock_response = Mock()
        mock_response.text = mock_html
        mock_get.return_value = mock_response
        
        # Test scraping
        article_url = "https://www.churchofjesuschrist.org/study/liahona/2008/01/empty-article"
        content = self.scraper.scrape_article(article_url)
        
        # Should return empty string for no meaningful content
        self.assertEqual(content, "")
    
    @patch('church_scraper.ContentScraper.robust_get')
    def test_scrape_article_http_error(self, mock_get):
        """Test scraping article with HTTP error."""
        # Mock HTTP error
        mock_get.side_effect = RequestException("HTTP 500 Server Error")
        
        # Test scraping should raise exception
        article_url = "https://www.churchofjesuschrist.org/study/liahona/2008/01/error-article"
        with self.assertRaises(RequestException):
            self.scraper.scrape_article(article_url)
    
    @patch('church_scraper.LiahonaScraper.scrape_article')
    @patch('church_scraper.LiahonaScraper.scrape_monthly_page')
    @patch('church_scraper.LiahonaScraper.get_monthly_urls')
    def test_run_success(self, mock_get_urls, mock_scrape_page, mock_scrape_article):
        """Test successful execution of complete Liahona scraping workflow."""
        # Mock monthly URLs
        mock_get_urls.return_value = [
            (2008, 1, "https://www.churchofjesuschrist.org/study/liahona/2008/01?lang=eng"),
            (2008, 2, "https://www.churchofjesuschrist.org/study/liahona/2008/02?lang=eng")
        ]
        
        # Mock article URLs from monthly pages
        mock_scrape_page.side_effect = [
            ["https://www.churchofjesuschrist.org/study/liahona/2008/01/article-1"],
            ["https://www.churchofjesuschrist.org/study/liahona/2008/02/article-2"]
        ]
        
        # Mock article content
        mock_scrape_article.return_value = "Sample article content for testing."
        
        # Run the scraper
        self.scraper.run()
        
        # Verify methods were called
        mock_get_urls.assert_called_once()
        self.assertEqual(mock_scrape_page.call_count, 2)
        self.assertEqual(mock_scrape_article.call_count, 2)
        
        # Verify files were created
        jan_dir = Path(self.temp_dir) / "liahona" / "2008-01"
        feb_dir = Path(self.temp_dir) / "liahona" / "2008-02"
        self.assertTrue(jan_dir.exists())
        self.assertTrue(feb_dir.exists())
    
    @patch('church_scraper.LiahonaScraper.scrape_monthly_page')
    @patch('church_scraper.LiahonaScraper.get_monthly_urls')
    def test_run_no_articles_found(self, mock_get_urls, mock_scrape_page):
        """Test run method when no articles are found on monthly pages."""
        # Mock monthly URLs
        mock_get_urls.return_value = [
            (2008, 1, "https://www.churchofjesuschrist.org/study/liahona/2008/01?lang=eng")
        ]
        
        # Mock empty article URLs
        mock_scrape_page.return_value = []
        
        # Run the scraper (should complete without errors)
        self.scraper.run()
        
        # Verify methods were called
        mock_get_urls.assert_called_once()
        mock_scrape_page.assert_called_once()
    
    @patch('church_scraper.LiahonaScraper.scrape_article')
    @patch('church_scraper.LiahonaScraper.scrape_monthly_page')
    @patch('church_scraper.LiahonaScraper.get_monthly_urls')
    def test_run_skip_existing_files(self, mock_get_urls, mock_scrape_page, mock_scrape_article):
        """Test run method skips existing files for resumable operations."""
        # Create existing file
        jan_dir = Path(self.temp_dir) / "liahona" / "2008-01"
        jan_dir.mkdir(parents=True, exist_ok=True)
        existing_file = jan_dir / "article-1.txt"
        existing_file.write_text("Existing content", encoding='utf-8')
        
        # Mock monthly URLs
        mock_get_urls.return_value = [
            (2008, 1, "https://www.churchofjesuschrist.org/study/liahona/2008/01?lang=eng")
        ]
        
        # Mock article URLs (including existing one)
        mock_scrape_page.return_value = [
            "https://www.churchofjesuschrist.org/study/liahona/2008/01/article-1",  # Existing
            "https://www.churchofjesuschrist.org/study/liahona/2008/01/article-2"   # New
        ]
        
        # Mock article content
        mock_scrape_article.return_value = "New article content."
        
        # Run the scraper
        self.scraper.run()
        
        # Should only scrape the new article (not the existing one)
        mock_scrape_article.assert_called_once_with(
            "https://www.churchofjesuschrist.org/study/liahona/2008/01/article-2"
        )
        
        # Existing file should remain unchanged
        self.assertEqual(existing_file.read_text(encoding='utf-8'), "Existing content")
    
    @patch('church_scraper.LiahonaScraper.get_monthly_urls')
    def test_run_handles_exceptions(self, mock_get_urls):
        """Test run method handles exceptions gracefully."""
        # Mock URLs that will cause an exception
        mock_get_urls.side_effect = Exception("Test exception")
        
        # Run should raise the exception
        with self.assertRaises(Exception):
            self.scraper.run()


if __name__ == '__main__':
    unittest.main()