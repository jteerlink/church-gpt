#!/usr/bin/env python3
"""
Unit tests for the Church Content Scraper.

Tests the HTTP session management, error handling, and retry mechanisms
of the ContentScraper base class.
"""

import pytest
import requests
import time
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    ReadTimeout,
    RequestException,
    SSLError,
    Timeout
)

from src.church_scraper import ScraperConfig, ContentScraper


class TestContentScraper:
    """Test cases for ContentScraper base class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = ScraperConfig(
            start_year=2020,
            end_year=2021,
            delay=0.1  # Shorter delay for tests
        )
        self.scraper = ContentScraper(self.config)
    
    def test_setup_session(self):
        """Test that HTTP session is properly configured."""
        session = self.scraper.setup_session()
        
        # Check that it's a requests.Session instance
        assert isinstance(session, requests.Session)
        
        # Check headers are set correctly
        assert session.headers['User-Agent'] == self.config.user_agent
        assert 'Accept' in session.headers
        assert 'Accept-Language' in session.headers
        
        # Check that adapters are mounted
        assert 'http://' in session.adapters
        assert 'https://' in session.adapters
    
    @patch('src.church_scraper.core.requests.Session.get')
    def test_robust_get_success(self, mock_get):
        """Test successful HTTP request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Make request
        response = self.scraper.robust_get('http://example.com')
        
        # Verify response
        assert response == mock_response
        mock_get.assert_called_once()
    
    @patch('src.church_scraper.core.requests.Session.get')
    def test_robust_get_404_error(self, mock_get):
        """Test handling of 404 errors."""
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        # Should raise HTTPError for 404
        with pytest.raises(HTTPError):
            self.scraper.robust_get('http://example.com/notfound')
    
    @patch('src.church_scraper.core.requests.Session.get')
    @patch('src.church_scraper.core.time.sleep')
    def test_robust_get_retry_on_connection_error(self, mock_sleep, mock_get):
        """Test retry mechanism for connection errors."""
        # Mock connection error on first two attempts, success on third
        mock_get.side_effect = [
            ConnectionError("Connection failed"),
            ConnectionError("Connection failed"),
            Mock(status_code=200, raise_for_status=Mock())
        ]
        
        # Make request
        response = self.scraper.robust_get('http://example.com')
        
        # Verify retries occurred
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 3  # Sleep called: config delay + 2 retry delays
        assert response.status_code == 200
    
    @patch('src.church_scraper.core.requests.Session.get')
    @patch('src.church_scraper.core.time.sleep')
    def test_robust_get_retry_on_ssl_error(self, mock_sleep, mock_get):
        """Test retry mechanism for SSL errors."""
        # Mock SSL error on first attempt, success on second
        mock_get.side_effect = [
            SSLError("SSL handshake failed"),
            Mock(status_code=200, raise_for_status=Mock())
        ]
        
        # Make request
        response = self.scraper.robust_get('http://example.com')
        
        # Verify retry occurred
        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 2  # Sleep called: config delay + 1 retry delay
        assert response.status_code == 200
    
    @patch('src.church_scraper.core.requests.Session.get')
    @patch('src.church_scraper.core.time.sleep')
    def test_robust_get_retry_on_timeout(self, mock_sleep, mock_get):
        """Test retry mechanism for timeout errors."""
        # Mock timeout error on first attempt, success on second
        mock_get.side_effect = [
            ReadTimeout("Request timed out"),
            Mock(status_code=200, raise_for_status=Mock())
        ]
        
        # Make request
        response = self.scraper.robust_get('http://example.com')
        
        # Verify retry occurred
        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 2  # Sleep called: config delay + 1 retry delay
        assert response.status_code == 200
    
    @patch('src.church_scraper.core.requests.Session.get')
    @patch('src.church_scraper.core.time.sleep')
    def test_robust_get_retry_on_500_error(self, mock_sleep, mock_get):
        """Test retry mechanism for 500 server errors."""
        # Mock 500 error on first attempt, success on second
        mock_response_error = Mock()
        mock_response_error.status_code = 500
        http_error = HTTPError("500 Server Error")
        http_error.response = mock_response_error
        mock_response_error.raise_for_status.side_effect = http_error
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status.return_value = None
        
        mock_get.side_effect = [mock_response_error, mock_response_success]
        
        # Make request
        response = self.scraper.robust_get('http://example.com')
        
        # Verify retry occurred
        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 2  # Sleep called: config delay + 1 retry delay
        assert response.status_code == 200
    
    @patch('src.church_scraper.core.requests.Session.get')
    @patch('src.church_scraper.core.time.sleep')
    def test_robust_get_max_retries_exceeded(self, mock_sleep, mock_get):
        """Test that RequestException is raised when max retries exceeded."""
        # Mock connection error for all attempts
        mock_get.side_effect = ConnectionError("Connection failed")
        
        # Should raise RequestException after max retries
        with pytest.raises(RequestException, match="Failed to fetch URL after 10 attempts"):
            self.scraper.robust_get('http://example.com')
        
        # Verify all retries were attempted
        assert mock_get.call_count == 10
        assert mock_sleep.call_count == 10  # Sleep called: config delay + 9 retry delays
    
    @patch('src.church_scraper.core.requests.Session.get')
    def test_robust_get_permanent_error_no_retry(self, mock_get):
        """Test that permanent errors (403) are not retried."""
        # Mock 403 response
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = HTTPError("403 Forbidden")
        mock_get.return_value = mock_response
        
        # Should raise HTTPError immediately without retries
        with pytest.raises(HTTPError):
            self.scraper.robust_get('http://example.com')
        
        # Verify no retries occurred
        assert mock_get.call_count == 1
    
    @patch('src.church_scraper.core.time.sleep')
    def test_robust_get_respects_delay_config(self, mock_sleep):
        """Test that configured delay is respected between requests."""
        with patch.object(self.scraper.session, 'get') as mock_get:
            # Mock connection error on first attempt, success on second
            mock_get.side_effect = [
                ConnectionError("Connection failed"),
                Mock(status_code=200, raise_for_status=Mock())
            ]
            
            # Make request
            self.scraper.robust_get('http://example.com')
            
            # Verify delay was called (once for retry, once for config delay)
            assert mock_sleep.call_count == 2
    
    def test_robust_get_default_timeout(self):
        """Test that default timeout is set when not provided."""
        with patch.object(self.scraper.session, 'get') as mock_get:
            mock_get.return_value = Mock(status_code=200, raise_for_status=Mock())
            
            # Make request without timeout
            self.scraper.robust_get('http://example.com')
            
            # Verify timeout was set
            args, kwargs = mock_get.call_args
            assert 'timeout' in kwargs
            assert kwargs['timeout'] == (10, 30)
    
    def test_robust_get_custom_timeout(self):
        """Test that custom timeout is preserved."""
        with patch.object(self.scraper.session, 'get') as mock_get:
            mock_get.return_value = Mock(status_code=200, raise_for_status=Mock())
            
            # Make request with custom timeout
            custom_timeout = (5, 15)
            self.scraper.robust_get('http://example.com', timeout=custom_timeout)
            
            # Verify custom timeout was used
            args, kwargs = mock_get.call_args
            assert kwargs['timeout'] == custom_timeout
    
    def test_extract_text_basic_html(self):
        """Test text extraction from basic HTML."""
        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Title</h1>
                <p>First paragraph with some text.</p>
                <p>Second paragraph with more text.</p>
            </body>
        </html>
        """
        
        text = self.scraper.extract_text(html)
        
        # Check that text is extracted and cleaned
        assert "Test Page" in text
        assert "Main Title" in text
        assert "First paragraph with some text." in text
        assert "Second paragraph with more text." in text
        
        # Check that HTML tags are removed
        assert "<html>" not in text
        assert "<p>" not in text
    
    def test_extract_text_removes_scripts_and_styles(self):
        """Test that script and style elements are removed."""
        html = """
        <html>
            <head>
                <style>body { color: red; }</style>
                <script>console.log('test');</script>
            </head>
            <body>
                <p>Visible content</p>
                <script>alert('popup');</script>
            </body>
        </html>
        """
        
        text = self.scraper.extract_text(html)
        
        # Check that visible content is preserved
        assert "Visible content" in text
        
        # Check that script and style content is removed
        assert "color: red" not in text
        assert "console.log" not in text
        assert "alert" not in text
    
    def test_extract_text_handles_whitespace(self):
        """Test that whitespace is properly normalized."""
        html = """
        <html>
            <body>
                <p>Line with    multiple   spaces</p>
                <p>
                    Line with
                    line breaks
                </p>
                <p></p>
                <p>   </p>
                <p>Final line</p>
            </body>
        </html>
        """
        
        text = self.scraper.extract_text(html)
        
        # Check that multiple spaces are normalized
        assert "multiple   spaces" not in text
        assert "multiple spaces" in text
        
        # Check that empty paragraphs don't create extra newlines
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Should have content from non-empty paragraphs
        assert any("multiple spaces" in line for line in non_empty_lines)
        assert any("line breaks" in line for line in non_empty_lines)
        assert any("Final line" in line for line in non_empty_lines)
    
    def test_extract_text_handles_malformed_html(self):
        """Test that malformed HTML is handled gracefully."""
        html = "<p>Unclosed paragraph<div>Nested content</p></div>"
        
        text = self.scraper.extract_text(html)
        
        # Should extract text even from malformed HTML
        assert "Unclosed paragraph" in text
        assert "Nested content" in text
    
    def test_extract_text_empty_html(self):
        """Test extraction from empty HTML."""
        html = "<html><body></body></html>"
        
        text = self.scraper.extract_text(html)
        
        # Should return empty string for empty HTML
        assert text == ""
    
    def test_extract_text_handles_exceptions(self):
        """Test that exceptions during text extraction are handled."""
        # Pass invalid input that might cause BeautifulSoup to fail
        invalid_html = None
        
        text = self.scraper.extract_text(invalid_html)
        
        # Should return empty string on error
        assert text == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])