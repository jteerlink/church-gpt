#!/usr/bin/env python3
"""
Tests for progress tracking and logging functionality in the Church Content Scraper.
"""

import io
import logging
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.church_scraper import (
    ProgressTracker,
    RateLimiter,
    setup_logging
)


class TestProgressTracker(unittest.TestCase):
    """Test cases for the ProgressTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Capture stdout for testing progress output
        self.stdout_capture = io.StringIO()
        self.original_stdout = sys.stdout
        
    def tearDown(self):
        """Clean up after tests."""
        sys.stdout = self.original_stdout
    
    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker(100, "Test Operation")
        
        self.assertEqual(tracker.total_items, 100)
        self.assertEqual(tracker.description, "Test Operation")
        self.assertEqual(tracker.processed_items, 0)
        self.assertEqual(tracker.skipped_items, 0)
        self.assertEqual(tracker.failed_items, 0)
        self.assertIsInstance(tracker.start_time, float)
    
    def test_progress_tracker_update(self):
        """Test progress tracking updates."""
        # Redirect stdout to capture progress output
        sys.stdout = self.stdout_capture
        
        tracker = ProgressTracker(10, "Test Progress")
        
        # Test initial update
        tracker.update(processed=2, current_item="item1")
        self.assertEqual(tracker.processed_items, 2)
        
        # Test cumulative updates
        tracker.update(processed=1, skipped=2, current_item="item2")
        self.assertEqual(tracker.processed_items, 3)
        self.assertEqual(tracker.skipped_items, 2)
        
        # Test failed items
        tracker.update(failed=1, current_item="item3")
        self.assertEqual(tracker.failed_items, 1)
        
        # Check that progress output was generated
        output = self.stdout_capture.getvalue()
        self.assertIn("Test Progress:", output)
        self.assertIn("60.0%", output)  # (3+2+1)/10 * 100
        self.assertIn("Processed: 3", output)
        self.assertIn("Skipped: 2", output)
        self.assertIn("Failed: 1", output)
    
    def test_progress_tracker_finish(self):
        """Test progress tracker completion."""
        # Redirect stdout to capture finish output
        sys.stdout = self.stdout_capture
        
        tracker = ProgressTracker(5, "Test Finish")
        tracker.update(processed=3, skipped=1, failed=1)
        
        # Mock time to test elapsed time calculation
        with patch('time.time') as mock_time:
            mock_time.return_value = tracker.start_time + 60  # 1 minute elapsed
            tracker.finish()
        
        output = self.stdout_capture.getvalue()
        self.assertIn("Test Finish completed!", output)
        self.assertIn("Items processed: 3", output)
        self.assertIn("Items skipped: 1", output)
        self.assertIn("Items failed: 1", output)
        self.assertIn("1m 0s", output)  # Should show 1 minute elapsed
    
    def test_progress_tracker_time_formatting(self):
        """Test time formatting in progress tracker."""
        tracker = ProgressTracker(1, "Test Time")
        
        # Test seconds formatting
        self.assertEqual(tracker._format_time(30.5), "30.5s")
        
        # Test minutes formatting
        self.assertEqual(tracker._format_time(90), "1m 30s")
        
        # Test hours formatting
        self.assertEqual(tracker._format_time(3661), "1h 1m")
    
    def test_progress_tracker_eta_calculation(self):
        """Test ETA calculation in progress tracker."""
        # Redirect stdout to capture ETA output
        sys.stdout = self.stdout_capture
        
        tracker = ProgressTracker(10, "Test ETA")
        
        # Mock time to control elapsed time
        with patch('time.time') as mock_time:
            # Start time
            start_time = 1000.0
            tracker.start_time = start_time
            
            # After processing 2 items in 10 seconds
            mock_time.return_value = start_time + 10.0
            tracker.update(processed=2, current_item="test")
            
            output = self.stdout_capture.getvalue()
            # Should calculate ETA based on 5 seconds per item for remaining 8 items
            self.assertIn("ETA:", output)


class TestRateLimiter(unittest.TestCase):
    """Test cases for the RateLimiter class."""
    
    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(2.0)
        
        self.assertEqual(limiter.delay, 2.0)
        self.assertEqual(limiter.last_request_time, 0)
    
    def test_rate_limiter_wait_no_delay_needed(self):
        """Test rate limiter when no delay is needed."""
        limiter = RateLimiter(1.0)
        
        # Mock time to simulate time passage
        with patch('time.time') as mock_time, patch('time.sleep') as mock_sleep:
            # First request - should not wait
            mock_time.return_value = 1000.0
            limiter.wait(show_progress=False)
            mock_sleep.assert_not_called()
            
            # Second request after sufficient time - should not wait
            mock_time.return_value = 1002.0  # 2 seconds later
            limiter.wait(show_progress=False)
            mock_sleep.assert_not_called()
    
    def test_rate_limiter_wait_with_delay(self):
        """Test rate limiter when delay is needed."""
        limiter = RateLimiter(2.0)
        
        with patch('time.time') as mock_time, patch('time.sleep') as mock_sleep:
            # First request
            mock_time.return_value = 1000.0
            limiter.wait(show_progress=False)
            
            # Second request too soon - should wait
            mock_time.return_value = 1001.0  # Only 1 second later
            limiter.wait(show_progress=False)
            
            # Should sleep for approximately 1 second (2.0 - 1.0)
            mock_sleep.assert_called_once()
            sleep_time = mock_sleep.call_args[0][0]
            self.assertAlmostEqual(sleep_time, 1.0, places=1)
    
    def test_rate_limiter_progress_display(self):
        """Test rate limiter progress display for long waits."""
        limiter = RateLimiter(2.0)
        
        # Capture stdout for progress output
        stdout_capture = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = stdout_capture
        
        try:
            with patch('time.time') as mock_time, patch('time.sleep'):
                # First request
                mock_time.return_value = 1000.0
                limiter.wait(show_progress=False)
                
                # Second request requiring a long wait (should show progress)
                mock_time.return_value = 1000.1  # Very soon after
                limiter._show_wait_progress(1.0)  # Test progress display directly
                
                output = stdout_capture.getvalue()
                self.assertIn("Rate limiting:", output)
                self.assertIn("100%", output)
        
        finally:
            sys.stdout = original_stdout


class TestLoggingSetup(unittest.TestCase):
    """Test cases for logging setup functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()
        
    def tearDown(self):
        """Clean up after tests."""
        # Reset logging configuration
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.WARNING)
    
    def test_setup_logging_creates_log_directory(self):
        """Test that setup_logging creates logs directory."""
        # Change to temp directory for this test
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            setup_logging(verbose=False)
            
            log_dir = Path(self.temp_dir) / "logs"
            self.assertTrue(log_dir.exists())
            self.assertTrue(log_dir.is_dir())
        finally:
            os.chdir(original_cwd)
    
    def test_setup_logging_creates_log_file(self):
        """Test that setup_logging creates a log file."""
        # Change to temp directory for this test
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            setup_logging(verbose=False)
            
            log_dir = Path(self.temp_dir) / "logs"
            log_files = list(log_dir.glob("church_scraper_*.log"))
            self.assertEqual(len(log_files), 1)
            
            # Test that we can write to the log file
            logger = logging.getLogger("test")
            logger.info("Test log message")
            
            # Check that the message was written to the file
            with open(log_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn("Test log message", content)
        finally:
            os.chdir(original_cwd)
    
    def test_setup_logging_verbose_mode(self):
        """Test logging setup in verbose mode."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            setup_logging(verbose=True)
            
            # Check that root logger is set to DEBUG level
            root_logger = logging.getLogger()
            self.assertEqual(root_logger.level, logging.DEBUG)
        finally:
            os.chdir(original_cwd)
    
    def test_setup_logging_normal_mode(self):
        """Test logging setup in normal mode."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            setup_logging(verbose=False)
            
            # Check that root logger is set to INFO level
            root_logger = logging.getLogger()
            self.assertEqual(root_logger.level, logging.INFO)
        finally:
            os.chdir(original_cwd)
    
    def test_logging_output_format(self):
        """Test that logging output has correct format."""
        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        logger = logging.getLogger("test_format")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log a test message
        logger.info("Test formatting message")
        
        output = log_capture.getvalue()
        self.assertIn("INFO: Test formatting message", output)
    
    def test_external_library_logging_suppressed(self):
        """Test that external library logging is suppressed."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            setup_logging(verbose=False)
            
            # Check that urllib3 and requests loggers are set to WARNING
            urllib3_logger = logging.getLogger('urllib3')
            requests_logger = logging.getLogger('requests')
            
            self.assertEqual(urllib3_logger.level, logging.WARNING)
            self.assertEqual(requests_logger.level, logging.WARNING)
        finally:
            os.chdir(original_cwd)


class TestProgressIntegration(unittest.TestCase):
    """Integration tests for progress tracking with scraping operations."""
    
    def test_progress_with_mock_scraping(self):
        """Test progress tracking with simulated scraping operations."""
        # Capture stdout for progress output
        stdout_capture = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = stdout_capture
        
        try:
            # Simulate a scraping operation with progress tracking
            total_items = 5
            progress = ProgressTracker(total_items, "Mock Scraping")
            
            # Simulate processing items with different outcomes
            for i in range(total_items):
                if i == 0:
                    # Successful processing
                    progress.update(processed=1, current_item=f"item_{i}")
                elif i == 1:
                    # Skipped item
                    progress.update(skipped=1, current_item=f"item_{i}")
                elif i == 2:
                    # Failed item
                    progress.update(failed=1, current_item=f"item_{i}")
                else:
                    # More successful processing
                    progress.update(processed=1, current_item=f"item_{i}")
                
                # Small delay to simulate processing time
                time.sleep(0.01)
            
            progress.finish()
            
            output = stdout_capture.getvalue()
            
            # Verify progress output contains expected elements
            self.assertIn("Mock Scraping:", output)
            self.assertIn("100.0%", output)  # Progress shows as 100.0%
            self.assertIn("Processed: 3", output)
            self.assertIn("Skipped: 1", output)
            self.assertIn("Failed: 1", output)
            self.assertIn("Mock Scraping completed!", output)
        
        finally:
            sys.stdout = original_stdout
    
    def test_rate_limiting_with_progress(self):
        """Test rate limiting integration with progress feedback."""
        limiter = RateLimiter(0.1)  # Very short delay for testing
        
        # Capture stdout for progress output
        stdout_capture = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = stdout_capture
        
        try:
            # Simulate multiple requests with rate limiting
            start_time = time.time()
            
            for i in range(3):
                limiter.wait(show_progress=False)  # Don't show progress for short delays
            
            elapsed_time = time.time() - start_time
            
            # Should take at least 0.2 seconds (2 delays of 0.1s each)
            self.assertGreaterEqual(elapsed_time, 0.15)
        
        finally:
            sys.stdout = original_stdout


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)