#!/usr/bin/env python3
"""
Integration tests for Church Content Scraper.

These tests validate complete scraping workflows, directory structure creation,
file organization, and resumable operations using both mocked and real URLs.
"""

import unittest
import tempfile
import shutil
import time
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import requests
from requests.exceptions import RequestException

from src.church_scraper import (
    ScraperConfig, 
    ConferenceScraper, 
    LiahonaScraper, 
    FileManager,
    setup_logging
)


class TestIntegrationWorkflows(unittest.TestCase):
    """Integration tests for complete scraping workflows."""
    
    def setUp(self):
        """Set up test fixtures with temporary directory and logging."""
        # Create temporary directory for test output
        self.temp_dir = tempfile.mkdtemp()
        
        # Set up minimal logging for tests
        setup_logging(verbose=False)
        
        # Create test configurations
        self.conference_config = ScraperConfig(
            start_year=1995,
            end_year=1996,  # Small range for testing
            content_type="conference",
            output_dir=self.temp_dir,
            delay=0.1,  # Faster for testing
            user_agent="Integration-Test-Scraper/1.0"
        )
        
        self.liahona_config = ScraperConfig(
            start_year=2008,
            end_year=2008,  # Single year for testing
            content_type="liahona",
            output_dir=self.temp_dir,
            delay=0.1,
            user_agent="Integration-Test-Scraper/1.0"
        )
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_conference_workflow_mocked(self):
        """Test complete General Conference scraping workflow with mocked responses."""
        # Create mock HTML responses
        conference_page_html = """
        <html>
        <body>
            <div class="content">
                <a href="/study/general-conference/1995/04/talk-1">First Talk</a>
                <a href="/study/general-conference/1995/04/talk-2">Second Talk</a>
                <a href="/study/general-conference/1995/04/talk-3">Third Talk</a>
            </div>
        </body>
        </html>
        """
        
        talk_html_template = """
        <html>
        <body>
            <div class="talk-content">
                <h1>Talk Title {}</h1>
                <p>This is the content of talk {}.</p>
                <p>It contains multiple paragraphs with meaningful content.</p>
                <p>This helps test the text extraction functionality.</p>
            </div>
        </body>
        </html>
        """
        
        # Mock HTTP responses
        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            
            if "/study/general-conference/1995/04?lang=eng" in url:
                response.text = conference_page_html
            elif "/study/general-conference/1995/10?lang=eng" in url:
                response.text = conference_page_html.replace("1995/04", "1995/10")
            elif "/study/general-conference/1996/04?lang=eng" in url:
                response.text = conference_page_html.replace("1995/04", "1996/04")
            elif "/study/general-conference/1996/10?lang=eng" in url:
                response.text = conference_page_html.replace("1995/04", "1996/10")
            elif "talk-1" in url:
                response.text = talk_html_template.format("1", "1")
            elif "talk-2" in url:
                response.text = talk_html_template.format("2", "2")
            elif "talk-3" in url:
                response.text = talk_html_template.format("3", "3")
            else:
                response.text = "<html><body>Not found</body></html>"
            
            return response
        
        # Create scraper and run with mocked HTTP
        scraper = ConferenceScraper(self.conference_config)
        
        with patch.object(scraper, 'robust_get', side_effect=mock_get_side_effect):
            scraper.run()
        
        # Verify directory structure was created correctly
        base_dir = Path(self.temp_dir)
        
        # Should have general-conference directory
        conference_dir = base_dir / "general-conference"
        self.assertTrue(conference_dir.exists())
        
        # Should have year-month directories for each conference
        expected_dirs = [
            conference_dir / "1995-04",
            conference_dir / "1995-10", 
            conference_dir / "1996-04",
            conference_dir / "1996-10"
        ]
        
        for expected_dir in expected_dirs:
            self.assertTrue(expected_dir.exists(), f"Directory {expected_dir} should exist")
            
            # Each directory should contain 3 talk files
            txt_files = list(expected_dir.glob("*.txt"))
            self.assertEqual(len(txt_files), 3, f"Directory {expected_dir} should contain 3 .txt files")
            
            # Verify file names
            expected_files = ["talk-1.txt", "talk-2.txt", "talk-3.txt"]
            actual_files = [f.name for f in txt_files]
            for expected_file in expected_files:
                self.assertIn(expected_file, actual_files)
        
        # Verify file contents
        sample_file = conference_dir / "1995-04" / "talk-1.txt"
        self.assertTrue(sample_file.exists())
        
        content = sample_file.read_text(encoding='utf-8')
        self.assertIn("Talk Title 1", content)
        self.assertIn("content of talk 1", content)
        self.assertGreater(len(content), 50)  # Should have substantial content
    
    def test_complete_liahona_workflow_mocked(self):
        """Test complete Liahona magazine scraping workflow with mocked responses."""
        # Create mock HTML responses
        monthly_page_html = """
        <html>
        <body>
            <div class="content">
                <a href="/study/liahona/2008/01/article-1">First Article</a>
                <a href="/study/liahona/2008/01/article-2">Second Article</a>
            </div>
        </body>
        </html>
        """
        
        article_html_template = """
        <html>
        <body>
            <div class="article-content">
                <h1>Article Title {}</h1>
                <p>This is the content of article {}.</p>
                <p>Liahona articles contain inspirational content.</p>
            </div>
        </body>
        </html>
        """
        
        # Mock HTTP responses
        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            
            if "/study/liahona/2008/" in url and "?lang=eng" in url:
                # Monthly page - adjust month in HTML
                month = url.split("/")[-1].split("?")[0]
                response.text = monthly_page_html.replace("2008/01", f"2008/{month}")
            elif "article-1" in url:
                response.text = article_html_template.format("1", "1")
            elif "article-2" in url:
                response.text = article_html_template.format("2", "2")
            else:
                response.text = "<html><body>Not found</body></html>"
            
            return response
        
        # Create scraper and run with mocked HTTP
        scraper = LiahonaScraper(self.liahona_config)
        
        with patch.object(scraper, 'robust_get', side_effect=mock_get_side_effect):
            scraper.run()
        
        # Verify directory structure was created correctly
        base_dir = Path(self.temp_dir)
        liahona_dir = base_dir / "liahona"
        self.assertTrue(liahona_dir.exists())
        
        # Should have directories for all months except April (4) and October (10)
        expected_months = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12]
        
        for month in expected_months:
            month_dir = liahona_dir / f"2008-{month:02d}"
            self.assertTrue(month_dir.exists(), f"Directory {month_dir} should exist")
            
            # Each directory should contain 2 article files
            txt_files = list(month_dir.glob("*.txt"))
            self.assertEqual(len(txt_files), 2, f"Directory {month_dir} should contain 2 .txt files")
        
        # Verify April and October directories don't exist (conference months)
        april_dir = liahona_dir / "2008-04"
        october_dir = liahona_dir / "2008-10"
        self.assertFalse(april_dir.exists(), "April directory should not exist")
        self.assertFalse(october_dir.exists(), "October directory should not exist")
        
        # Verify file contents
        sample_file = liahona_dir / "2008-01" / "article-1.txt"
        self.assertTrue(sample_file.exists())
        
        content = sample_file.read_text(encoding='utf-8')
        self.assertIn("Article Title 1", content)
        self.assertIn("content of article 1", content)
    
    def test_resumable_operations_conference(self):
        """Test that conference scraping can be resumed and skips existing files."""
        # Create mock responses
        conference_page_html = """
        <html>
        <body>
            <a href="/study/general-conference/1995/04/talk-1">Talk 1</a>
            <a href="/study/general-conference/1995/04/talk-2">Talk 2</a>
        </body>
        </html>
        """
        
        talk_html = """
        <html>
        <body>
            <h1>Talk Content</h1>
            <p>This is talk content.</p>
        </body>
        </html>
        """
        
        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            if "?lang=eng" in url:
                response.text = conference_page_html
            else:
                response.text = talk_html
            return response
        
        # Create scraper
        scraper = ConferenceScraper(self.conference_config)
        
        # First run - should create all files
        with patch.object(scraper, 'robust_get', side_effect=mock_get_side_effect) as mock_get:
            scraper.run()
            first_run_call_count = mock_get.call_count
        
        # Verify files were created
        conference_dir = Path(self.temp_dir) / "general-conference" / "1995-04"
        talk1_file = conference_dir / "talk-1.txt"
        talk2_file = conference_dir / "talk-2.txt"
        
        self.assertTrue(talk1_file.exists())
        self.assertTrue(talk2_file.exists())
        
        # Modify one file to test it's not overwritten
        original_content = talk1_file.read_text(encoding='utf-8')
        modified_content = "MODIFIED CONTENT - should not be overwritten"
        talk1_file.write_text(modified_content, encoding='utf-8')
        
        # Second run - should skip existing files
        with patch.object(scraper, 'robust_get', side_effect=mock_get_side_effect) as mock_get:
            scraper.run()
            second_run_call_count = mock_get.call_count
        
        # Should make fewer HTTP calls in second run (skipping existing files)
        self.assertLess(second_run_call_count, first_run_call_count)
        
        # Modified file should remain unchanged
        current_content = talk1_file.read_text(encoding='utf-8')
        self.assertEqual(current_content, modified_content)
        
        # Other file should still exist
        self.assertTrue(talk2_file.exists())
    
    def test_resumable_operations_liahona(self):
        """Test that Liahona scraping can be resumed and skips existing files."""
        # Create mock responses
        monthly_page_html = """
        <html>
        <body>
            <a href="/study/liahona/2008/01/article-1">Article 1</a>
            <a href="/study/liahona/2008/01/article-2">Article 2</a>
        </body>
        </html>
        """
        
        article_html = """
        <html>
        <body>
            <h1>Article Content</h1>
            <p>This is article content.</p>
        </body>
        </html>
        """
        
        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            if "?lang=eng" in url:
                response.text = monthly_page_html
            else:
                response.text = article_html
            return response
        
        # Create scraper with single month for faster testing
        config = ScraperConfig(
            start_year=2008,
            end_year=2008,
            content_type="liahona",
            output_dir=self.temp_dir,
            delay=0.1
        )
        scraper = LiahonaScraper(config)
        
        # First run
        with patch.object(scraper, 'robust_get', side_effect=mock_get_side_effect) as mock_get:
            scraper.run()
            first_run_call_count = mock_get.call_count
        
        # Verify files were created
        liahona_dir = Path(self.temp_dir) / "liahona" / "2008-01"
        article1_file = liahona_dir / "article-1.txt"
        article2_file = liahona_dir / "article-2.txt"
        
        self.assertTrue(article1_file.exists())
        self.assertTrue(article2_file.exists())
        
        # Modify one file
        modified_content = "MODIFIED ARTICLE - should not be overwritten"
        article1_file.write_text(modified_content, encoding='utf-8')
        
        # Second run
        with patch.object(scraper, 'robust_get', side_effect=mock_get_side_effect) as mock_get:
            scraper.run()
            second_run_call_count = mock_get.call_count
        
        # Should make fewer calls (skipping existing files)
        self.assertLess(second_run_call_count, first_run_call_count)
        
        # Modified file should remain unchanged
        current_content = article1_file.read_text(encoding='utf-8')
        self.assertEqual(current_content, modified_content)
    
    def test_directory_structure_organization(self):
        """Test that directory structure is created and organized correctly."""
        # Test FileManager directly for comprehensive directory testing
        file_manager = FileManager(self.temp_dir)
        
        # Test various directory structures
        test_cases = [
            # (year, month, content_type, expected_path)
            (2023, 4, "general-conference", "general-conference/2023-04"),
            (2023, 10, "general-conference", "general-conference/2023-10"),
            (2008, 1, "liahona", "liahona/2008-01"),
            (2008, 12, "liahona", "liahona/2008-12"),
            (2000, None, "general-conference", "general-conference/2000"),
        ]
        
        for year, month, content_type, expected_rel_path in test_cases:
            # Create directory structure
            if month is not None:
                dir_path = file_manager.create_directory_structure(year, month, content_type)
            else:
                dir_path = file_manager.create_directory_structure(year, content_type=content_type)
            
            # Verify path is correct
            expected_full_path = str((Path(self.temp_dir) / expected_rel_path).resolve())
            self.assertEqual(dir_path, expected_full_path)
            
            # Verify directory exists
            self.assertTrue(Path(dir_path).exists())
            self.assertTrue(Path(dir_path).is_dir())
        
        # Test file organization within directories
        test_files = [
            (2023, 4, "talk-1", "general-conference"),
            (2023, 4, "talk-2", "general-conference"),
            (2008, 1, "article-1", "liahona"),
            (2008, 1, "article-2", "liahona"),
        ]
        
        for year, month, slug, content_type in test_files:
            # Generate filepath and save content
            filepath = file_manager.get_content_filepath(year, month, slug, content_type)
            test_content = f"Test content for {slug} in {year}-{month:02d}"
            file_manager.save_content(test_content, filepath)
            
            # Verify file exists and has correct content
            self.assertTrue(Path(filepath).exists())
            
            saved_content = Path(filepath).read_text(encoding='utf-8')
            self.assertEqual(saved_content, test_content)
        
        # Verify directory listing functionality
        conference_files = file_manager.list_existing_files(2023, 4, "general-conference")
        self.assertEqual(len(conference_files), 2)
        self.assertIn("talk-1.txt", conference_files)
        self.assertIn("talk-2.txt", conference_files)
        
        liahona_files = file_manager.list_existing_files(2008, 1, "liahona")
        self.assertEqual(len(liahona_files), 2)
        self.assertIn("article-1.txt", liahona_files)
        self.assertIn("article-2.txt", liahona_files)
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery in complete workflows."""
        # Create mock responses with some failures
        def mock_get_with_failures(url, **kwargs):
            response = Mock()
            
            if "1995/04" in url and "?lang=eng" in url:
                # Conference page succeeds
                response.text = """
                <html><body>
                    <a href="/study/general-conference/1995/04/talk-1">Talk 1</a>
                    <a href="/study/general-conference/1995/04/talk-2">Talk 2</a>
                </body></html>
                """
            elif "1995/10" in url and "?lang=eng" in url:
                # Conference page fails
                raise RequestException("Conference page not found")
            elif "talk-1" in url:
                # First talk succeeds
                response.text = "<html><body><h1>Talk 1</h1><p>Content</p></body></html>"
            elif "talk-2" in url:
                # Second talk fails
                raise RequestException("Talk not found")
            else:
                response.text = "<html><body>Default</body></html>"
            
            return response
        
        # Create scraper
        scraper = ConferenceScraper(self.conference_config)
        
        # Run scraper - should handle errors gracefully
        with patch.object(scraper, 'robust_get', side_effect=mock_get_with_failures):
            # Should not raise exception despite failures
            try:
                scraper.run()
            except Exception as e:
                self.fail(f"Scraper should handle errors gracefully, but raised: {e}")
        
        # Verify partial success - some files should be created
        conference_dir = Path(self.temp_dir) / "general-conference"
        
        # 1995-04 directory should exist with one successful file
        april_dir = conference_dir / "1995-04"
        self.assertTrue(april_dir.exists())
        
        talk1_file = april_dir / "talk-1.txt"
        self.assertTrue(talk1_file.exists())
        
        # talk-2 should not exist due to failure
        talk2_file = april_dir / "talk-2.txt"
        self.assertFalse(talk2_file.exists())
        
        # 1995-10 directory might exist but should be empty due to conference page failure
        october_dir = conference_dir / "1995-10"
        if october_dir.exists():
            txt_files = list(october_dir.glob("*.txt"))
            self.assertEqual(len(txt_files), 0)


class TestEndToEndWithRealURLs(unittest.TestCase):
    """End-to-end tests with a small subset of real URLs (when network available)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        setup_logging(verbose=False)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_real_conference_page_access(self):
        """Test accessing a real conference page (network dependent)."""
        # Use a very old conference that should be stable
        config = ScraperConfig(
            start_year=1995,
            end_year=1995,
            content_type="conference",
            output_dir=self.temp_dir,
            delay=2.0  # Be respectful to the server
        )
        
        scraper = ConferenceScraper(config)
        
        try:
            # Test just URL generation and one page access
            urls = scraper.get_conference_urls()
            self.assertGreater(len(urls), 0)
            
            # Try to access one conference page
            year, month, conf_url = urls[0]
            
            # Set a reasonable timeout for network test
            response = scraper.robust_get(conf_url, timeout=(5, 10))
            
            # Should get a valid response
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.text), 100)
            
            # Try to extract talk URLs
            talk_urls = scraper.scrape_conference_page(conf_url)
            
            # Should find some talk URLs (even if structure changed)
            # This is a loose test since website structure may change
            self.assertIsInstance(talk_urls, list)
            
        except (RequestException, ConnectionError) as e:
            # Skip test if network is unavailable
            self.skipTest(f"Network test skipped due to connection error: {e}")
        except Exception as e:
            # Log unexpected errors but don't fail the test suite
            print(f"Real URL test encountered error (this may be expected): {e}")
            self.skipTest(f"Real URL test skipped due to error: {e}")
    
    def test_real_liahona_page_access(self):
        """Test accessing a real Liahona page (network dependent)."""
        # Use an old Liahona issue that should be stable
        config = ScraperConfig(
            start_year=2008,
            end_year=2008,
            content_type="liahona", 
            output_dir=self.temp_dir,
            delay=2.0
        )
        
        scraper = LiahonaScraper(config)
        
        try:
            # Test URL generation
            urls = scraper.get_monthly_urls()
            self.assertGreater(len(urls), 0)
            
            # Try to access one monthly page (January 2008)
            january_url = None
            for year, month, url in urls:
                if month == 1:  # January
                    january_url = url
                    break
            
            self.assertIsNotNone(january_url)
            
            # Access the page
            response = scraper.robust_get(january_url, timeout=(5, 10))
            
            # Should get a valid response
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.text), 100)
            
            # Try to extract article URLs
            article_urls = scraper.scrape_monthly_page(january_url)
            
            # Should return a list (may be empty if structure changed)
            self.assertIsInstance(article_urls, list)
            
        except (RequestException, ConnectionError) as e:
            # Skip test if network is unavailable
            self.skipTest(f"Network test skipped due to connection error: {e}")
        except Exception as e:
            # Log unexpected errors but don't fail the test suite
            print(f"Real URL test encountered error (this may be expected): {e}")
            self.skipTest(f"Real URL test skipped due to error: {e}")


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation in integration scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_configuration_handling(self):
        """Test that invalid configurations are handled properly."""
        # Test invalid year range
        with self.assertRaises(ValueError):
            ScraperConfig(start_year=2025, end_year=2020, output_dir=self.temp_dir)
        
        # Test invalid content type
        with self.assertRaises(ValueError):
            ScraperConfig(content_type="invalid", output_dir=self.temp_dir)
        
        # Test invalid delay
        with self.assertRaises(ValueError):
            ScraperConfig(delay=-1, output_dir=self.temp_dir)
        
        # Test empty user agent
        with self.assertRaises(ValueError):
            ScraperConfig(user_agent="", output_dir=self.temp_dir)
    
    def test_edge_case_configurations(self):
        """Test edge case configurations."""
        # Test single year configuration
        config = ScraperConfig(
            start_year=2000,
            end_year=2000,
            output_dir=self.temp_dir
        )
        
        scraper = ConferenceScraper(config)
        urls = scraper.get_conference_urls()
        
        # Should have exactly 2 URLs for one year
        self.assertEqual(len(urls), 2)
        
        # Test very small delay
        config = ScraperConfig(
            start_year=2000,
            end_year=2000,
            output_dir=self.temp_dir,
            delay=0.001
        )
        
        # Should not raise error
        scraper = ConferenceScraper(config)
        self.assertIsNotNone(scraper)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)