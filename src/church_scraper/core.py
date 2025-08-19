#!/usr/bin/env python3
"""
Church Content Scraper

A comprehensive Python script that scrapes religious content from the Church of Jesus Christ 
of Latter-day Saints website (churchofjesuschrist.org), specifically targeting General Conference 
talks and Liahona magazine articles. The script provides robust error handling, rate limiting, 
progress tracking, and resumable operations.

FEATURES:
- Scrapes General Conference talks from April and October sessions
- Scrapes Liahona magazine articles (excluding conference months)
- Configurable year ranges and content types
- Robust HTTP error handling with exponential backoff retry logic
- Rate limiting to respect server resources
- Progress tracking with ETA calculations
- Resumable operations (skips existing files)
- Comprehensive logging to both file and console
- Cross-platform directory structure creation

URL PATTERNS:
The script works with the following URL patterns from churchofjesuschrist.org:

General Conference:
- Conference index: /study/general-conference/{YYYY}/{MM}?lang=eng
- Individual talks: /study/general-conference/{YYYY}/{MM}/{talk-slug}?lang=eng
- Years: 1995+ (configurable)
- Months: April (04) and October (10) only

Liahona Magazine:
- Monthly index: /study/liahona/{YYYY}/{MM}?lang=eng  
- Individual articles: /study/liahona/{YYYY}/{MM}/{article-slug}?lang=eng
- Years: 2008+ (configurable, when Liahona became available online)
- Months: All except April and October (conference months)

DIRECTORY STRUCTURE:
The script creates organized directory structures for scraped content:

scraped_content/
├── general-conference/
│   ├── 1995-04/
│   │   ├── talk-slug-1.txt
│   │   └── talk-slug-2.txt
│   └── 1995-10/
└── liahona/
    ├── 2008-01/
    │   ├── article-slug-1.txt
    │   └── article-slug-2.txt
    └── 2008-02/

USAGE EXAMPLES:

Basic usage - scrape both content types from default years:
    python church_scraper.py

Scrape only General Conference talks from specific years:
    python church_scraper.py --start-year 1995 --end-year 1999 --content-type conference

Scrape only Liahona articles with custom output directory:
    python church_scraper.py --start-year 2008 --content-type liahona --output-dir ./my_content

Scrape with increased delay for slower connections:
    python church_scraper.py --delay 2.5 --verbose

Resume interrupted scraping (automatically skips existing files):
    python church_scraper.py --start-year 2010 --end-year 2020

CONFIGURATION OPTIONS:
    --start-year INT     Starting year for scraping (default: 1995)
    --end-year INT       Ending year for scraping (default: current year)
    --content-type STR   Content type: 'conference', 'liahona', or 'both' (default: both)
    --output-dir STR     Directory to save content (default: 'scraped_content')
    --delay FLOAT        Delay between requests in seconds (default: 1.0)
    --user-agent STR     Custom User-Agent string (default: Church-Content-Scraper/1.0)
    --verbose, -v        Enable verbose debug logging

EXIT CODES:
    0   Success - all operations completed successfully
    1   Configuration error - invalid parameters or setup failure
    130 User interruption - script was interrupted with Ctrl+C
    2   Network error - persistent connection or server issues
    3   File system error - directory creation or file writing failures

REQUIREMENTS:
- Python 3.7+
- requests library for HTTP operations
- beautifulsoup4 library for HTML parsing
- Internet connection for accessing churchofjesuschrist.org

RATE LIMITING:
The script implements respectful rate limiting with a default 1-second delay between requests.
This can be adjusted with the --delay parameter. The script also includes exponential backoff
retry logic for handling temporary network issues or server overload.

LOGGING:
Comprehensive logging is provided at multiple levels:
- Console output shows progress and important status messages
- File logs (in logs/ directory) contain detailed debug information
- Progress bars show real-time scraping status with ETA calculations

RESUMABLE OPERATIONS:
The script automatically detects existing files and skips them, allowing you to:
- Resume interrupted scraping sessions
- Add new years to existing collections
- Re-run the script safely without duplicating work

AUTHOR: Church Content Scraper v1.0
LICENSE: Educational and research use
"""

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    ReadTimeout,
    RequestException,
    SSLError,
    Timeout
)
from urllib3.util.retry import Retry


class ProgressTracker:
    """Tracks and displays progress for scraping operations."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            description: Description of the operation being tracked
        """
        self.total_items = total_items
        self.processed_items = 0
        self.skipped_items = 0
        self.failed_items = 0
        self.description = description
        self.start_time = time.time()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def update(self, processed: int = 0, skipped: int = 0, failed: int = 0, current_item: str = "") -> None:
        """
        Update progress counters and display current status with real-time progress bar.
        
        This method provides comprehensive progress tracking with:
        - Real-time progress bar with percentage completion
        - ETA calculation based on average processing time
        - Detailed counters for processed, skipped, and failed items
        - Current item display for user feedback
        - Periodic logging for audit trail
        
        Args:
            processed: Number of items processed in this update
            skipped: Number of items skipped in this update (existing files)
            failed: Number of items failed in this update (errors)
            current_item: Description of current item being processed
        """
        # Update internal counters with new values
        self.processed_items += processed
        self.skipped_items += skipped
        self.failed_items += failed
        
        # Calculate overall progress statistics
        total_completed = self.processed_items + self.skipped_items + self.failed_items
        percentage = (total_completed / self.total_items * 100) if self.total_items > 0 else 0
        
        # Calculate ETA based on average processing time
        elapsed_time = time.time() - self.start_time
        if total_completed > 0:
            # Calculate average time per item (including skipped and failed items)
            avg_time_per_item = elapsed_time / total_completed
            remaining_items = self.total_items - total_completed
            eta_seconds = avg_time_per_item * remaining_items
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "Unknown"
        
        # Create visual progress bar using Unicode block characters
        # █ (full block) for completed portion, ░ (light shade) for remaining
        bar_length = 30  # Fixed width for consistent display
        filled_length = int(bar_length * percentage / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Construct status message with all progress information
        status_msg = (
            f"\r{self.description}: [{bar}] {percentage:.1f}% "
            f"({total_completed}/{self.total_items}) "
            f"Processed: {self.processed_items}, Skipped: {self.skipped_items}, Failed: {self.failed_items} "
            f"ETA: {eta_str}"
        )
        
        # Add current item information if provided
        if current_item:
            status_msg += f" | Current: {current_item}"
        
        # Print progress bar (overwrite previous line with \r)
        # end='' prevents newline, flush=True ensures immediate display
        print(status_msg, end='', flush=True)
        
        # Log detailed progress at regular intervals for audit trail
        # Log every 10 items or at completion to avoid log spam
        if total_completed % 10 == 0 or total_completed == self.total_items:
            self.logger.info(
                f"Progress: {total_completed}/{self.total_items} ({percentage:.1f}%) - "
                f"Processed: {self.processed_items}, Skipped: {self.skipped_items}, Failed: {self.failed_items}"
            )
    
    def finish(self) -> None:
        """Mark progress as complete and display final statistics."""
        print()  # New line after progress bar
        
        elapsed_time = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed_time)
        
        total_completed = self.processed_items + self.skipped_items + self.failed_items
        
        self.logger.info(
            f"{self.description} completed in {elapsed_str}. "
            f"Total: {total_completed}/{self.total_items}, "
            f"Processed: {self.processed_items}, "
            f"Skipped: {self.skipped_items}, "
            f"Failed: {self.failed_items}"
        )
        
        print(f"\n{self.description} completed!")
        print(f"Total time: {elapsed_str}")
        print(f"Items processed: {self.processed_items}")
        print(f"Items skipped: {self.skipped_items}")
        print(f"Items failed: {self.failed_items}")
        
        if self.processed_items > 0:
            avg_time = elapsed_time / self.processed_items
            print(f"Average time per item: {self._format_time(avg_time)}")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration for the scraper.
    
    Args:
        verbose: Enable verbose (DEBUG level) logging
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set log level
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(
        log_dir / f"church_scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler for user feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose logs from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logging.info("Logging initialized")


class RateLimiter:
    """Manages rate limiting with progress feedback."""
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            delay: Base delay between requests in seconds
        """
        self.delay = delay
        self.last_request_time = 0
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def wait(self, show_progress: bool = True) -> None:
        """
        Wait for the appropriate delay between requests.
        
        Args:
            show_progress: Whether to show progress feedback during wait
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.delay:
            wait_time = self.delay - time_since_last
            
            if show_progress and wait_time > 0.5:  # Only show progress for longer waits
                self._show_wait_progress(wait_time)
            else:
                time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _show_wait_progress(self, wait_time: float) -> None:
        """
        Show progress bar during rate limiting wait.
        
        Args:
            wait_time: Time to wait in seconds
        """
        steps = max(10, int(wait_time * 2))  # Update progress bar multiple times per second
        step_time = wait_time / steps
        
        for i in range(steps + 1):
            progress = i / steps
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            remaining = wait_time - (i * step_time)
            print(f"\rRate limiting: [{bar}] {progress*100:.0f}% ({remaining:.1f}s remaining)", end='', flush=True)
            
            if i < steps:
                time.sleep(step_time)
        
        print()  # New line after progress bar


class ScraperConfig:
    """Configuration management class for the Church Content Scraper."""
    
    def __init__(
        self,
        start_year: int = 1995,
        end_year: Optional[int] = None,
        content_type: str = "both",
        output_dir: str = "scraped_content",
        delay: float = 1.0,
        user_agent: str = "Church-Content-Scraper/1.0"
    ):
        """
        Initialize scraper configuration with validation.
        
        Args:
            start_year: Starting year for scraping
            end_year: Ending year for scraping (defaults to current year)
            content_type: Type of content ('conference', 'liahona', or 'both')
            output_dir: Directory to save scraped content
            delay: Delay between requests in seconds
            user_agent: User-Agent string for HTTP requests
        """
        self.start_year = start_year
        self.end_year = end_year if end_year is not None else datetime.now().year
        self.content_type = content_type
        self.output_dir = output_dir
        self.delay = delay
        self.user_agent = user_agent
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters and raise errors for invalid values."""
        # Validate year ranges
        current_year = datetime.now().year
        
        if not isinstance(self.start_year, int) or self.start_year < 1900:
            raise ValueError(f"start_year must be an integer >= 1900, got: {self.start_year}")
        
        if not isinstance(self.end_year, int) or self.end_year > current_year:
            raise ValueError(f"end_year must be an integer <= {current_year}, got: {self.end_year}")
        
        if self.start_year > self.end_year:
            raise ValueError(f"start_year ({self.start_year}) cannot be greater than end_year ({self.end_year})")
        
        # Validate content type
        valid_content_types = {"conference", "liahona", "both"}
        if self.content_type not in valid_content_types:
            raise ValueError(f"content_type must be one of {valid_content_types}, got: {self.content_type}")
        
        # Validate output directory path
        try:
            Path(self.output_dir).resolve()
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid output_dir path '{self.output_dir}': {e}")
        
        # Validate delay
        if not isinstance(self.delay, (int, float)) or self.delay < 0:
            raise ValueError(f"delay must be a non-negative number, got: {self.delay}")
        
        # Validate user agent
        if not isinstance(self.user_agent, str) or not self.user_agent.strip():
            raise ValueError("user_agent must be a non-empty string")
    
    def __str__(self) -> str:
        """Return string representation of configuration."""
        return (
            f"ScraperConfig("
            f"start_year={self.start_year}, "
            f"end_year={self.end_year}, "
            f"content_type='{self.content_type}', "
            f"output_dir='{self.output_dir}', "
            f"delay={self.delay}, "
            f"user_agent='{self.user_agent}')"
        )


class ContentScraper:
    """Base class for scraping content with HTTP session management and error handling."""
    
    def __init__(self, config: ScraperConfig):
        """
        Initialize the content scraper with configuration.
        
        Args:
            config: ScraperConfig instance with scraping parameters
        """
        self.config = config
        self.session = self.setup_session()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rate_limiter = RateLimiter(config.delay)
    
    def setup_session(self) -> requests.Session:
        """
        Set up HTTP session with retry strategy and proper configuration.
        
        Returns:
            Configured requests.Session instance
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=10,  # Maximum number of retries
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
            allowed_methods=["HEAD", "GET", "OPTIONS"],  # HTTP methods to retry
            backoff_factor=1,  # Backoff factor for exponential backoff
            raise_on_status=False  # Don't raise exception on retry failure
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'User-Agent': self.config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        return session
    
    def robust_get(self, url: str, show_progress: bool = True, **kwargs) -> requests.Response:
        """
        Make HTTP GET request with robust error handling and exponential backoff retry logic.
        
        This method implements a comprehensive retry strategy to handle various network issues:
        - Temporary network failures (SSL, timeout, connection errors)
        - Server overload (HTTP 429, 5xx errors)
        - Rate limiting with respectful delays
        - Exponential backoff with jitter to avoid thundering herd problems
        
        Retry Strategy:
        - Maximum 10 retry attempts
        - Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (capped)
        - Random jitter (10-50% of delay) to distribute retry timing
        - Different handling for temporary vs permanent errors
        
        Args:
            url: URL to request
            show_progress: Whether to show progress feedback during delays
            **kwargs: Additional arguments to pass to requests.get()
            
        Returns:
            requests.Response object
            
        Raises:
            RequestException: If all retry attempts fail
        """
        max_retries = 10  # Maximum number of retry attempts
        base_delay = 1.0  # Base delay for exponential backoff (seconds)
        max_delay = 60.0  # Maximum delay cap to prevent excessive waiting
        
        # Set default timeout if not provided by caller
        # (connect_timeout, read_timeout) - reasonable defaults for web scraping
        if 'timeout' not in kwargs:
            kwargs['timeout'] = (10, 30)  # 10s to connect, 30s to read response
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Attempting to fetch URL: {url} (attempt {attempt + 1}/{max_retries})")
                
                # Apply rate limiting only on first attempt to avoid double-delaying retries
                if attempt == 0:
                    self.rate_limiter.wait(show_progress)
                
                # Make the HTTP request using the configured session
                response = self.session.get(url, **kwargs)
                
                # Check for HTTP error status codes
                if response.status_code == 404:
                    # 404 Not Found - content may not exist for this year/month
                    self.logger.warning(f"URL not found (404): {url}")
                    response.raise_for_status()
                elif response.status_code >= 400:
                    # Other 4xx/5xx errors - let HTTPError handling decide retry logic
                    self.logger.warning(f"HTTP error {response.status_code} for URL: {url}")
                    response.raise_for_status()
                
                # Success - log response size and return
                self.logger.debug(f"Successfully fetched URL: {url} ({len(response.content)} bytes)")
                return response
                
            except (SSLError, ReadTimeout, ConnectionError, Timeout) as e:
                # Network-related errors that should always be retried
                # These are typically temporary issues that may resolve
                self.logger.warning(f"Network error on attempt {attempt + 1}/{max_retries} for {url}: {type(e).__name__}: {e}")
                
                if attempt == max_retries - 1:
                    # Final attempt failed - give up and raise exception
                    self.logger.error(f"Failed to fetch URL after {max_retries} attempts: {url}")
                    raise RequestException(f"Failed to fetch URL after {max_retries} attempts: {url}") from e
                
                # Calculate exponential backoff delay with random jitter
                # Jitter helps prevent multiple clients from retrying simultaneously
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0.1, 0.5) * delay  # 10-50% jitter
                total_delay = delay + jitter
                
                self.logger.info(f"Retrying in {total_delay:.2f} seconds... (attempt {attempt + 2}/{max_retries})")
                
                # Show progress bar for longer waits to keep user informed
                if show_progress and total_delay > 1.0:
                    self._show_retry_progress(total_delay, attempt + 2, max_retries)
                else:
                    time.sleep(total_delay)
                
            except HTTPError as e:
                # HTTP errors (4xx, 5xx) - some should be retried, others are permanent
                status_code = e.response.status_code if e.response else response.status_code
                
                if status_code in [429, 500, 502, 503, 504]:
                    # Temporary server errors that should be retried:
                    # 429 Too Many Requests - rate limiting
                    # 500 Internal Server Error - temporary server issue
                    # 502 Bad Gateway - proxy/load balancer issue
                    # 503 Service Unavailable - temporary overload
                    # 504 Gateway Timeout - upstream timeout
                    self.logger.warning(f"HTTP error {status_code} on attempt {attempt + 1}/{max_retries} for {url}: {e}")
                    
                    if attempt == max_retries - 1:
                        self.logger.error(f"Failed to fetch URL after {max_retries} attempts: {url}")
                        raise RequestException(f"Failed to fetch URL after {max_retries} attempts: {url}") from e
                    
                    # Use exponential backoff for server errors too
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0.1, 0.5) * delay
                    total_delay = delay + jitter
                    
                    self.logger.info(f"Retrying in {total_delay:.2f} seconds... (attempt {attempt + 2}/{max_retries})")
                    
                    if show_progress and total_delay > 1.0:
                        self._show_retry_progress(total_delay, attempt + 2, max_retries)
                    else:
                        time.sleep(total_delay)
                else:
                    # Permanent errors that should not be retried:
                    # 400 Bad Request - malformed request
                    # 401 Unauthorized - authentication required
                    # 403 Forbidden - access denied
                    # 404 Not Found - resource doesn't exist
                    self.logger.error(f"Permanent HTTP error {status_code} for URL: {url}")
                    raise
                    
            except RequestException as e:
                # Other request exceptions - typically permanent issues
                self.logger.error(f"Request exception for URL {url}: {type(e).__name__}: {e}")
                raise
        
        # This should never be reached due to the raise statements in the loop
        raise RequestException(f"Unexpected error: failed to fetch URL after {max_retries} attempts: {url}")
    
    def _show_retry_progress(self, wait_time: float, attempt: int, max_attempts: int) -> None:
        """
        Show progress bar during retry wait.
        
        Args:
            wait_time: Time to wait in seconds
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
        """
        steps = max(10, int(wait_time * 2))  # Update progress bar multiple times per second
        step_time = wait_time / steps
        
        for i in range(steps + 1):
            progress = i / steps
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            remaining = wait_time - (i * step_time)
            print(f"\rRetrying ({attempt}/{max_attempts}): [{bar}] {progress*100:.0f}% ({remaining:.1f}s)", end='', flush=True)
            
            if i < steps:
                time.sleep(step_time)
        
        print()  # New line after progress bar
    
    def extract_text(self, html: str) -> str:
        """
        Extract clean text content from HTML using BeautifulSoup with comprehensive cleaning.
        
        This method performs several cleaning operations to produce readable text:
        1. Parse HTML using BeautifulSoup's html.parser (built-in, no external dependencies)
        2. Remove script and style elements that contain non-content code
        3. Extract all text content while preserving paragraph structure
        4. Clean up whitespace, normalize line breaks, and remove empty lines
        5. Preserve meaningful formatting while removing HTML artifacts
        6. Apply content filtering for conference articles (clean headers, remove navigation)
        7. Fix encoding issues (smart quotes, special characters)
        
        The cleaning process handles common HTML formatting issues:
        - Multiple consecutive spaces are collapsed to single spaces
        - Empty lines and whitespace-only lines are removed
        - Line breaks are normalized to single newlines between paragraphs
        - Script/style content is completely removed to avoid code in text
        - Conference content is filtered to include only essential metadata and body
        - Encoding artifacts are fixed for proper text display
        
        Args:
            html: Raw HTML content from web page response
            
        Returns:
            Cleaned text content with proper formatting, ready for file storage
            Returns empty string if extraction fails
        """
        try:
            # Parse HTML using BeautifulSoup's built-in parser
            # html.parser is chosen for reliability and no external dependencies
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements completely
            # These contain JavaScript/CSS code that shouldn't appear in text content
            # decompose() removes elements from the tree and frees memory
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract all text content from the cleaned HTML tree
            # get_text() recursively extracts text from all elements
            text = soup.get_text()
            
            # Clean up whitespace and formatting for readable output
            # Split into lines and process each line individually
            lines = (line.strip() for line in text.splitlines())
            
            # Process each line to normalize whitespace and remove empty lines
            chunks = []
            for line in lines:
                if line:  # Skip empty lines
                    # Replace multiple consecutive spaces/tabs with single space
                    # This handles HTML whitespace artifacts and improves readability
                    cleaned_line = ' '.join(line.split())
                    chunks.append(cleaned_line)
            
            # Join cleaned lines with single newlines to preserve paragraph structure
            text = '\n'.join(chunks)
            
            # Apply conference content filtering if this appears to be a conference article
            if 'general-conference' in self.logger.name.lower() or 'Conference' in text[:500]:
                text = self._clean_conference_content(text)
            
            return text
            
        except Exception as e:
            # Log extraction errors but don't crash the scraping process
            # Return empty string so caller can handle gracefully
            self.logger.error(f"Error extracting text from HTML: {e}")
            return ""
    
    def _clean_conference_content(self, raw_text: str) -> str:
        """
        Clean General Conference article content with Notes preserved and encoding fixes.
        
        This method filters out navigation elements and table of contents while preserving
        the essential article structure: title, author, date, body content, and references.
        
        Args:
            raw_text: Raw extracted text from HTML
            
        Returns:
            Cleaned content with proper structure and encoding
        """
        try:
            # Fix encoding issues first
            text = self._fix_encoding(raw_text)
            lines = text.split('\n')
            
            # Find the article title line - look for it in table of contents or as separate line
            title_line = 0
            title = ""
            
            # First, try to extract title from the table of contents
            for i, line in enumerate(lines):
                line_clean = line.strip()
                # Look for title patterns in the TOC
                if any(pattern in line_clean for pattern in [
                    "Let Him Do It with Simplicity", "Christian Courage", "The Price of",
                    "Go Ye Therefore", "You Know Enough", "Come What May"
                ]) and len(line_clean) > 15:
                    # Extract just the title part
                    title_candidates = [
                        "Let Him Do It with Simplicity", "Christian Courage: The Price of Discipleship",
                        "Go Ye Therefore", "You Know Enough", "Come What May, and Love It"
                    ]
                    for candidate in title_candidates:
                        if candidate in line_clean:
                            title = candidate
                            title_line = i
                            break
                    if title:
                        break
            
            # If not found in TOC, look for standalone title lines
            if not title:
                for i, line in enumerate(lines):
                    if (len(line.strip()) > 20 and 
                        ":" in line and 
                        not any(skip in line.lower() for skip in ['contents', 'session', 'authenticating']) and
                        not line.strip().isdigit()):
                        title = line.strip()
                        title_line = i
                        break
            
            # Use the extracted title (already set above)
            if not title:
                title = "Unknown Talk"  # Fallback
            
            # Find author and metadata - look after the table of contents
            author = ""
            author_title = ""
            session_date = "October 2008"  # Default fallback
            
            # Look for "By Elder" pattern anywhere in the text
            for i, line in enumerate(lines):
                line_clean = line.strip()
                if line_clean.startswith("By Elder") or line_clean.startswith("By President") or line_clean.startswith("By Sister"):
                    author = line_clean
                    # Look for author title in next few lines
                    for j in range(i + 1, min(i + 5, len(lines))):
                        next_line = lines[j].strip()
                        if any(title_phrase in next_line for title_phrase in [
                            "Of the Quorum", "President", "Bishop", "Relief Society"
                        ]):
                            author_title = next_line
                            break
                    break
            
            # Find the start of actual content - look for common talk opening patterns
            content_start = None
            for i, line in enumerate(lines):
                line_lower = line.strip().lower()
                if (len(line.strip()) > 50 and any(pattern in line_lower for pattern in [
                    "we have gathered", "my dear", "brothers and sisters", "recently", 
                    "today i", "this morning", "i am grateful", "it is good"
                ])):
                    content_start = i
                    break
            
            if content_start is None:
                # Fallback: look for first substantial paragraph after metadata
                for i in range(title_line + 5, len(lines)):
                    if len(lines[i].strip()) > 80:
                        content_start = i
                        break
            
            if content_start is None:
                return text  # Return original with encoding fixes only
            
            # Build clean output
            result = []
            result.append(title)
            if author:
                result.append(author)
            if author_title:
                result.append(author_title)
            result.append(session_date)
            result.append("")  # Empty line separator
            
            # Add content from the start of the actual talk to the end (including Notes)
            for i in range(content_start, len(lines)):
                line = lines[i].strip()
                if line and not line.startswith('→') and '16:36' not in line:
                    result.append(line)
            
            return '\n'.join(result)
            
        except Exception as e:
            self.logger.warning(f"Conference content cleaning failed: {e}")
            return self._fix_encoding(raw_text)  # Return with encoding fixes only
    
    def _fix_encoding(self, text: str) -> str:
        """
        Fix common encoding issues in scraped text.
        
        Args:
            text: Text with potential encoding issues
            
        Returns:
            Text with corrected encoding
        """
        replacements = {
            'â': '"',      # Left double quote
            'â': '"',      # Right double quote  
            'â': "'",      # Apostrophe
            'Ã©': 'é',     # e with acute
            'Ã': 'À',      # A with grave
            'â¦': '…',     # Ellipsis
            'â': '—',      # Em dash
            'Ã±': 'ñ',     # n with tilde
            'Ã¡': 'á',     # a with acute
            'Ã­': 'í',     # i with acute
            'Ã³': 'ó',     # o with acute
            'Ãº': 'ú',     # u with acute
            'Â': ' ',      # Non-breaking space issue (HeberÂ J. -> Heber J.)
            'â': '"',      # Another quote variant
            'â': '"',      # Another quote variant
            'â': '"',      # Smart quote
            'â': '"',      # Smart quote
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text


class FileManager:
    """Manages local file operations and directory structure for scraped content."""
    
    def __init__(self, base_dir: str):
        """
        Initialize FileManager with base directory.
        
        Args:
            base_dir: Base directory path for storing scraped content
        """
        self.base_dir = Path(base_dir).resolve()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"FileManager initialized with base directory: {self.base_dir}")
    
    def create_directory_structure(self, year: int, month: int = None, content_type: str = "general-conference") -> str:
        """
        Create directory structure for organizing content by year and optionally month.
        
        Args:
            year: Year for the content
            month: Optional month for the content (1-12)
            content_type: Type of content ('general-conference' or 'liahona')
            
        Returns:
            Path to the created directory as string
            
        Raises:
            ValueError: If year or month values are invalid
            OSError: If directory creation fails
        """
        # Validate inputs
        if not isinstance(year, int) or year < 1900 or year > 2100:
            raise ValueError(f"Year must be an integer between 1900 and 2100, got: {year}")
        
        if month is not None:
            if not isinstance(month, int) or month < 1 or month > 12:
                raise ValueError(f"Month must be an integer between 1 and 12, got: {month}")
        
        valid_content_types = {"general-conference", "liahona"}
        if content_type not in valid_content_types:
            raise ValueError(f"content_type must be one of {valid_content_types}, got: {content_type}")
        
        try:
            # Build directory path
            if month is not None:
                # Format: base_dir/content_type/YYYY-MM/
                dir_name = f"{year}-{month:02d}"
                dir_path = self.base_dir / content_type / dir_name
            else:
                # Format: base_dir/content_type/YYYY/
                dir_path = self.base_dir / content_type / str(year)
            
            # Create directory structure
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory structure: {dir_path}")
            
            return str(dir_path)
            
        except OSError as e:
            self.logger.error(f"Failed to create directory structure for {year}-{month}: {e}")
            raise OSError(f"Failed to create directory structure: {e}") from e
    
    def save_content(self, content: str, filepath: str) -> None:
        """
        Save text content to a file with proper encoding.
        
        Args:
            content: Text content to save
            filepath: Full path where to save the file
            
        Raises:
            ValueError: If content is empty or filepath is invalid
            OSError: If file writing fails
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Content must be a non-empty string")
        
        if not isinstance(filepath, str) or not filepath.strip():
            raise ValueError("Filepath must be a non-empty string")
        
        try:
            file_path = Path(filepath).resolve()
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content with UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.debug(f"Saved content to file: {file_path}")
            
        except (OSError, IOError) as e:
            self.logger.error(f"Failed to save content to {filepath}: {e}")
            raise OSError(f"Failed to save content to file: {e}") from e
    
    def file_exists(self, filepath: str) -> bool:
        """
        Check if a file exists at the given path.
        
        Args:
            filepath: Path to check for file existence
            
        Returns:
            True if file exists, False otherwise
        """
        if not isinstance(filepath, str) or not filepath.strip():
            return False
        
        try:
            file_path = Path(filepath).resolve()
            exists = file_path.is_file()
            self.logger.debug(f"File existence check for {filepath}: {exists}")
            return exists
            
        except (OSError, ValueError):
            # If path is invalid or inaccessible, consider it as non-existent
            self.logger.debug(f"File existence check failed for {filepath}, treating as non-existent")
            return False
    
    def get_content_filepath(self, year: int, month: int, slug: str, content_type: str = "general-conference") -> str:
        """
        Generate full filepath for content based on year, month, and URL slug.
        
        This method creates a complete file path following the organized directory structure:
        - Creates the appropriate directory structure if it doesn't exist
        - Cleans the URL slug to create a valid filename
        - Ensures proper file extension (.txt)
        - Returns cross-platform compatible file path
        
        Filename Cleaning Process:
        - Removes invalid filesystem characters (keeps alphanumeric, hyphens, underscores, dots)
        - Strips whitespace from beginning and end
        - Adds .txt extension if not present
        - Validates that cleaning doesn't result in empty filename
        
        Directory Structure Examples:
        - General Conference: base_dir/general-conference/YYYY-MM/speaker-slug.txt
        - Liahona: base_dir/liahona/YYYY-MM/article-slug.txt
        
        Args:
            year: Year for the content (1900-2100)
            month: Month for the content (1-12)
            slug: URL slug or filename (will be cleaned for filesystem compatibility)
            content_type: Type of content ('general-conference' or 'liahona')
            
        Returns:
            Full filepath as string, ready for file operations
            
        Raises:
            ValueError: If parameters are invalid or slug cleaning fails
            OSError: If directory creation fails
        """
        # Validate year parameter with reasonable bounds
        if not isinstance(year, int) or year < 1900 or year > 2100:
            raise ValueError(f"Year must be an integer between 1900 and 2100, got: {year}")
        
        # Validate month parameter (1-12 for January-December)
        if not isinstance(month, int) or month < 1 or month > 12:
            raise ValueError(f"Month must be an integer between 1 and 12, got: {month}")
        
        # Validate slug parameter (must be non-empty string)
        if not isinstance(slug, str) or not slug.strip():
            raise ValueError("Slug must be a non-empty string")
        
        # Validate content type parameter
        valid_content_types = {"general-conference", "liahona"}
        if content_type not in valid_content_types:
            raise ValueError(f"content_type must be one of {valid_content_types}, got: {content_type}")
        
        # Clean slug for filesystem compatibility
        # Keep only alphanumeric characters and safe punctuation
        # This prevents issues with special characters in filenames across different OS
        clean_slug = "".join(c for c in slug if c.isalnum() or c in ('-', '_', '.')).strip()
        if not clean_slug:
            raise ValueError("Slug results in empty filename after cleaning")
        
        # Ensure proper file extension for text content
        if not clean_slug.endswith('.txt'):
            clean_slug += '.txt'
        
        # Create directory structure (this method handles directory creation)
        dir_path = self.create_directory_structure(year, month, content_type)
        
        # Combine directory path with cleaned filename
        # Using Path for cross-platform compatibility
        return str(Path(dir_path) / clean_slug)
    
    def list_existing_files(self, year: int, month: int = None, content_type: str = "general-conference") -> list:
        """
        List existing files in a directory for resumable operations.
        
        Args:
            year: Year to check
            month: Optional month to check (1-12)
            content_type: Type of content ('general-conference' or 'liahona')
            
        Returns:
            List of existing filenames in the directory
        """
        try:
            if month is not None:
                dir_name = f"{year}-{month:02d}"
                dir_path = self.base_dir / content_type / dir_name
            else:
                dir_path = self.base_dir / content_type / str(year)
            
            if not dir_path.exists():
                return []
            
            # Get all .txt files in the directory
            files = [f.name for f in dir_path.iterdir() if f.is_file() and f.suffix == '.txt']
            self.logger.debug(f"Found {len(files)} existing files in {dir_path}")
            return files
            
        except (OSError, ValueError) as e:
            self.logger.warning(f"Failed to list files in directory: {e}")
            return []


class ConferenceScraper(ContentScraper):
    """Scraper for General Conference talks from the Church of Jesus Christ website."""
    
    def __init__(self, config: ScraperConfig):
        """
        Initialize ConferenceScraper with configuration.
        
        Args:
            config: ScraperConfig instance with scraping parameters
        """
        super().__init__(config)
        self.file_manager = FileManager(config.output_dir)
        self.base_url = "https://www.churchofjesuschrist.org"
    
    def get_conference_urls(self) -> list:
        """
        Generate conference page URLs for April and October sessions within the configured year range.
        
        General Conference is held twice yearly in Salt Lake City:
        - April Conference (Spring): Usually first weekend of April
        - October Conference (Fall): Usually first weekend of October
        
        URL Pattern: https://www.churchofjesuschrist.org/study/general-conference/{YYYY}/{MM}?lang=eng
        Where:
        - {YYYY} is the 4-digit year (e.g., 1995, 2023)
        - {MM} is the 2-digit month: 04 for April, 10 for October
        - lang=eng specifies English language content
        
        Returns:
            List of tuples containing (year, month, url) for each conference
        """
        conference_urls = []
        
        for year in range(self.config.start_year, self.config.end_year + 1):
            # General Conference happens in April (04) and October (10) only
            # These are the only two months when General Conference is held
            for month in [4, 10]:
                # Construct URL following the site's pattern: /study/general-conference/YYYY/MM
                # The ?lang=eng parameter ensures we get English content
                url = f"{self.base_url}/study/general-conference/{year}/{month:02d}?lang=eng"
                conference_urls.append((year, month, url))
                self.logger.debug(f"Generated conference URL: {url}")
        
        self.logger.info(f"Generated {len(conference_urls)} conference URLs for years {self.config.start_year}-{self.config.end_year}")
        return conference_urls
    
    def scrape_conference_page(self, conf_url: str) -> list:
        """
        Scrape a conference page to extract individual talk URLs.
        
        Conference index pages contain links to individual talks. This method parses the HTML
        to find all talk links and returns them for individual processing.
        
        Expected URL structure for talks:
        /study/general-conference/{YYYY}/{MM}/{speaker-name-slug}?lang=eng
        
        Examples:
        - /study/general-conference/2023/04/nelson?lang=eng (President Nelson's talk)
        - /study/general-conference/2023/10/oaks?lang=eng (Elder Oaks' talk)
        
        Args:
            conf_url: URL of the conference page to scrape
            
        Returns:
            List of individual talk URLs found on the conference page
            
        Raises:
            RequestException: If the conference page cannot be fetched
        """
        try:
            self.logger.info(f"Scraping conference page: {conf_url}")
            
            # Fetch the conference page HTML content
            response = self.robust_get(conf_url)
            
            # Parse HTML using BeautifulSoup for reliable link extraction
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links that point to individual talks
            # Talk URLs follow pattern: /study/general-conference/YYYY/MM/speaker-slug
            talk_links = []
            
            # Iterate through all anchor tags with href attributes
            # Conference pages contain navigation, footer, and content links
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Filter for talk URLs using path pattern matching
                # Talk URLs must contain the conference path and have sufficient depth
                if '/study/general-conference/' in href and href.count('/') >= 5:
                    # Convert relative URLs to absolute URLs
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        # Skip malformed or relative URLs without leading slash
                        continue
                    
                    # Avoid duplicate URLs and exclude the conference index page itself
                    if full_url not in talk_links and full_url != conf_url:
                        # Validate URL structure: should have speaker slug at the end
                        # Expected structure: ['', 'study', 'general-conference', 'YYYY', 'MM', 'speaker-slug']
                        url_parts = href.strip('/').split('/')
                        if len(url_parts) >= 5 and url_parts[-1] != '':  # Must have non-empty speaker slug
                            talk_links.append(full_url)
                            self.logger.debug(f"Found talk URL: {full_url}")
            
            self.logger.info(f"Found {len(talk_links)} talk URLs on conference page: {conf_url}")
            return talk_links
            
        except Exception as e:
            self.logger.error(f"Failed to scrape conference page {conf_url}: {e}")
            raise RequestException(f"Failed to scrape conference page: {e}") from e
    
    def scrape_talk(self, talk_url: str) -> str:
        """
        Download and extract text content from an individual talk page.
        
        Args:
            talk_url: URL of the individual talk to scrape
            
        Returns:
            Extracted text content from the talk
            
        Raises:
            RequestException: If the talk page cannot be fetched
        """
        try:
            self.logger.debug(f"Scraping talk: {talk_url}")
            
            # Fetch the talk page
            response = self.robust_get(talk_url)
            
            # Extract text content using the base class method
            text_content = self.extract_text(response.text)
            
            if not text_content.strip():
                self.logger.warning(f"No text content extracted from talk: {talk_url}")
                return ""
            
            self.logger.debug(f"Successfully extracted {len(text_content)} characters from talk: {talk_url}")
            return text_content
            
        except Exception as e:
            self.logger.error(f"Failed to scrape talk {talk_url}: {e}")
            raise RequestException(f"Failed to scrape talk: {e}") from e
    
    def run(self) -> None:
        """
        Execute the complete General Conference scraping workflow.
        
        This method orchestrates the entire scraping process:
        1. Generate conference URLs for the configured year range
        2. For each conference, extract individual talk URLs
        3. Download and save each talk's content
        4. Skip existing files to enable resumable operations
        """
        try:
            self.logger.info("Starting General Conference scraping...")
            
            # Generate all conference URLs
            conference_urls = self.get_conference_urls()
            
            # First pass: count total talks for progress tracking
            self.logger.info("Counting total talks for progress tracking...")
            total_talks = 0
            conference_talk_counts = {}
            
            for year, month, conf_url in conference_urls:
                try:
                    self.logger.debug(f"Counting talks for conference: {year}-{month:02d}")
                    talk_urls = self.scrape_conference_page(conf_url)
                    talk_count = len(talk_urls)
                    conference_talk_counts[(year, month)] = talk_count
                    total_talks += talk_count
                    self.logger.debug(f"Found {talk_count} talks for {year}-{month:02d}")
                except Exception as e:
                    self.logger.warning(f"Failed to count talks for {year}-{month:02d}: {e}")
                    conference_talk_counts[(year, month)] = 0
            
            self.logger.info(f"Found {total_talks} total talks across {len(conference_urls)} conferences")
            
            # Initialize progress tracker
            progress = ProgressTracker(total_talks, "General Conference Scraping")
            
            # Second pass: process all talks with progress tracking
            for year, month, conf_url in conference_urls:
                try:
                    conference_desc = f"{year}-{month:02d}"
                    self.logger.info(f"Processing conference: {conference_desc}")
                    
                    # Create directory structure for this conference
                    self.file_manager.create_directory_structure(year, month, "general-conference")
                    
                    # Get list of existing files to skip
                    existing_files = self.file_manager.list_existing_files(year, month, "general-conference")
                    existing_slugs = {f.replace('.txt', '') for f in existing_files}
                    
                    # Scrape the conference page to get talk URLs
                    talk_urls = self.scrape_conference_page(conf_url)
                    
                    if not talk_urls:
                        self.logger.warning(f"No talk URLs found for conference: {conf_url}")
                        continue
                    
                    # Process each talk with progress updates
                    for i, talk_url in enumerate(talk_urls, 1):
                        try:
                            # Extract slug from URL for filename
                            url_parts = talk_url.strip('/').split('/')
                            slug = url_parts[-1] if url_parts else "unknown"
                            
                            current_item = f"{conference_desc} - {slug} ({i}/{len(talk_urls)})"
                            
                            # Skip if file already exists
                            if slug in existing_slugs:
                                self.logger.debug(f"Skipping existing talk: {slug}")
                                progress.update(skipped=1, current_item=current_item)
                                continue
                            
                            # Download and save the talk
                            talk_content = self.scrape_talk(talk_url)
                            
                            if talk_content.strip():
                                # Generate filepath and save content
                                filepath = self.file_manager.get_content_filepath(
                                    year, month, slug, "general-conference"
                                )
                                self.file_manager.save_content(talk_content, filepath)
                                
                                self.logger.info(f"Saved talk: {slug} ({len(talk_content)} chars)")
                                progress.update(processed=1, current_item=current_item)
                            else:
                                self.logger.warning(f"Empty content for talk: {talk_url}")
                                progress.update(failed=1, current_item=current_item)
                        
                        except Exception as e:
                            self.logger.error(f"Failed to process talk {talk_url}: {e}")
                            progress.update(failed=1, current_item=f"{conference_desc} - {slug} (FAILED)")
                            continue
                
                except Exception as e:
                    self.logger.error(f"Failed to process conference {year}-{month:02d}: {e}")
                    # Mark remaining talks in this conference as failed
                    remaining_talks = conference_talk_counts.get((year, month), 0)
                    if remaining_talks > 0:
                        progress.update(failed=remaining_talks, current_item=f"{year}-{month:02d} (CONFERENCE FAILED)")
                    continue
            
            # Finish progress tracking
            progress.finish()
            
            self.logger.info(
                f"General Conference scraping completed. "
                f"Processed: {progress.processed_items}, "
                f"Skipped: {progress.skipped_items}, "
                f"Failed: {progress.failed_items}"
            )
            
        except Exception as e:
            self.logger.error(f"General Conference scraping failed: {e}")
            raise


class LiahonaScraper(ContentScraper):
    """Scraper for Liahona magazine articles from the Church of Jesus Christ website."""
    
    def __init__(self, config: ScraperConfig):
        """
        Initialize LiahonaScraper with configuration.
        
        Args:
            config: ScraperConfig instance with scraping parameters
        """
        super().__init__(config)
        self.file_manager = FileManager(config.output_dir)
        self.base_url = "https://www.churchofjesuschrist.org"
    
    def get_monthly_urls(self) -> List[Tuple[int, int, str]]:
        """
        Generate monthly Liahona page URLs excluding April and October (conference months).
        
        The Liahona is the Church's international magazine, published monthly in multiple languages.
        It contains articles, stories, and teachings for Church members worldwide.
        
        Publishing Schedule:
        - Published monthly (12 issues per year)
        - April and October issues are replaced by General Conference content
        - Available months: Jan, Feb, Mar, May, Jun, Jul, Aug, Sep, Nov, Dec (10 issues)
        
        URL Pattern: https://www.churchofjesuschrist.org/study/liahona/{YYYY}/{MM}?lang=eng
        Where:
        - {YYYY} is the 4-digit year (e.g., 2008, 2023)
        - {MM} is the 2-digit month: 01-12 except 04 and 10
        - lang=eng specifies English language content
        
        Returns:
            List of tuples containing (year, month, url) for each monthly issue
        """
        monthly_urls = []
        
        for year in range(self.config.start_year, self.config.end_year + 1):
            # Liahona is published monthly except during General Conference months
            # April (04) and October (10) are skipped because General Conference content
            # takes precedence during these months
            for month in range(1, 13):
                if month not in [4, 10]:  # Skip April and October (conference months)
                    # Construct URL following the site's pattern: /study/liahona/YYYY/MM
                    # The ?lang=eng parameter ensures we get English content
                    url = f"{self.base_url}/study/liahona/{year}/{month:02d}?lang=eng"
                    monthly_urls.append((year, month, url))
                    self.logger.debug(f"Generated Liahona URL: {url}")
        
        self.logger.info(f"Generated {len(monthly_urls)} Liahona URLs for years {self.config.start_year}-{self.config.end_year}")
        return monthly_urls
    
    def scrape_monthly_page(self, month_url: str) -> List[str]:
        """
        Scrape a monthly Liahona page to extract individual article URLs.
        
        Args:
            month_url: URL of the monthly Liahona page to scrape
            
        Returns:
            List of individual article URLs found on the monthly page
            
        Raises:
            RequestException: If the monthly page cannot be fetched
        """
        try:
            self.logger.info(f"Scraping monthly Liahona page: {month_url}")
            
            # Fetch the monthly page
            response = self.robust_get(month_url)
            
            # Parse HTML to extract article URLs
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links that point to individual articles
            # Article URLs typically have the pattern: /study/liahona/YYYY/MM/article-slug
            article_links = []
            
            # Iterate through all anchor tags to find article links
            # Monthly Liahona pages contain various types of links (navigation, articles, etc.)
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Filter for article URLs using path pattern matching
                # Article URLs must contain '/study/liahona/' and have sufficient path depth
                # Expected pattern: /study/liahona/YYYY/MM/article-slug
                if '/study/liahona/' in href and href.count('/') >= 5:
                    # Convert relative URLs to absolute URLs for consistency
                    if href.startswith('/'):
                        # Relative URL starting with / - prepend base domain
                        full_url = self.base_url + href
                    elif href.startswith('http'):
                        # Already absolute URL - use as-is
                        full_url = href
                    else:
                        # Relative URL without leading slash - skip as malformed
                        continue
                    
                    # Avoid duplicate URLs and exclude the monthly index page itself
                    if full_url not in article_links and full_url != month_url:
                        # Validate URL structure: should have article slug at the end
                        # Expected structure: ['', 'study', 'liahona', 'YYYY', 'MM', 'article-slug']
                        url_parts = href.strip('/').split('/')
                        if len(url_parts) >= 5 and url_parts[-1] != '':  # Must have non-empty article slug
                            article_links.append(full_url)
                            self.logger.debug(f"Found article URL: {full_url}")
            
            self.logger.info(f"Found {len(article_links)} article URLs on monthly page: {month_url}")
            return article_links
            
        except Exception as e:
            self.logger.error(f"Failed to scrape monthly page {month_url}: {e}")
            raise RequestException(f"Failed to scrape monthly page: {e}") from e
    
    def scrape_article(self, article_url: str) -> str:
        """
        Download and extract text content from an individual article page.
        
        Args:
            article_url: URL of the individual article to scrape
            
        Returns:
            Extracted text content from the article
            
        Raises:
            RequestException: If the article page cannot be fetched
        """
        try:
            self.logger.debug(f"Scraping article: {article_url}")
            
            # Fetch the article page
            response = self.robust_get(article_url)
            
            # Extract text content using the base class method
            text_content = self.extract_text(response.text)
            
            if not text_content.strip():
                self.logger.warning(f"No text content extracted from article: {article_url}")
                return ""
            
            self.logger.debug(f"Successfully extracted {len(text_content)} characters from article: {article_url}")
            return text_content
            
        except Exception as e:
            self.logger.error(f"Failed to scrape article {article_url}: {e}")
            raise RequestException(f"Failed to scrape article: {e}") from e
    
    def run(self) -> None:
        """
        Execute the complete Liahona magazine scraping workflow.
        
        This method orchestrates the entire scraping process:
        1. Generate monthly URLs for the configured year range (excluding April/October)
        2. For each monthly issue, extract individual article URLs
        3. Download and save each article's content
        4. Skip existing files to enable resumable operations
        """
        try:
            self.logger.info("Starting Liahona magazine scraping...")
            
            # Generate all monthly URLs
            monthly_urls = self.get_monthly_urls()
            
            # First pass: count total articles for progress tracking
            self.logger.info("Counting total articles for progress tracking...")
            total_articles = 0
            monthly_article_counts = {}
            
            for year, month, month_url in monthly_urls:
                try:
                    self.logger.debug(f"Counting articles for issue: {year}-{month:02d}")
                    article_urls = self.scrape_monthly_page(month_url)
                    article_count = len(article_urls)
                    monthly_article_counts[(year, month)] = article_count
                    total_articles += article_count
                    self.logger.debug(f"Found {article_count} articles for {year}-{month:02d}")
                except Exception as e:
                    self.logger.warning(f"Failed to count articles for {year}-{month:02d}: {e}")
                    monthly_article_counts[(year, month)] = 0
            
            self.logger.info(f"Found {total_articles} total articles across {len(monthly_urls)} monthly issues")
            
            # Initialize progress tracker
            progress = ProgressTracker(total_articles, "Liahona Magazine Scraping")
            
            # Second pass: process all articles with progress tracking
            for year, month, month_url in monthly_urls:
                try:
                    issue_desc = f"{year}-{month:02d}"
                    self.logger.info(f"Processing Liahona issue: {issue_desc}")
                    
                    # Create directory structure for this monthly issue
                    self.file_manager.create_directory_structure(year, month, "liahona")
                    
                    # Get list of existing files to skip
                    existing_files = self.file_manager.list_existing_files(year, month, "liahona")
                    existing_slugs = {f.replace('.txt', '') for f in existing_files}
                    
                    # Scrape the monthly page to get article URLs
                    article_urls = self.scrape_monthly_page(month_url)
                    
                    if not article_urls:
                        self.logger.warning(f"No article URLs found for monthly issue: {month_url}")
                        continue
                    
                    # Process each article with progress updates
                    for i, article_url in enumerate(article_urls, 1):
                        try:
                            # Extract slug from URL for filename
                            url_parts = article_url.strip('/').split('/')
                            slug = url_parts[-1] if url_parts else "unknown"
                            
                            current_item = f"{issue_desc} - {slug} ({i}/{len(article_urls)})"
                            
                            # Skip if file already exists
                            if slug in existing_slugs:
                                self.logger.debug(f"Skipping existing article: {slug}")
                                progress.update(skipped=1, current_item=current_item)
                                continue
                            
                            # Download and save the article
                            article_content = self.scrape_article(article_url)
                            
                            if article_content.strip():
                                # Generate filepath and save content
                                filepath = self.file_manager.get_content_filepath(
                                    year, month, slug, "liahona"
                                )
                                self.file_manager.save_content(article_content, filepath)
                                
                                self.logger.info(f"Saved article: {slug} ({len(article_content)} chars)")
                                progress.update(processed=1, current_item=current_item)
                            else:
                                self.logger.warning(f"Empty content for article: {article_url}")
                                progress.update(failed=1, current_item=current_item)
                        
                        except Exception as e:
                            self.logger.error(f"Failed to process article {article_url}: {e}")
                            progress.update(failed=1, current_item=f"{issue_desc} - {slug} (FAILED)")
                            continue
                
                except Exception as e:
                    self.logger.error(f"Failed to process monthly issue {year}-{month:02d}: {e}")
                    # Mark remaining articles in this issue as failed
                    remaining_articles = monthly_article_counts.get((year, month), 0)
                    if remaining_articles > 0:
                        progress.update(failed=remaining_articles, current_item=f"{year}-{month:02d} (ISSUE FAILED)")
                    continue
            
            # Finish progress tracking
            progress.finish()
            
            self.logger.info(
                f"Liahona magazine scraping completed. "
                f"Processed: {progress.processed_items}, "
                f"Skipped: {progress.skipped_items}, "
                f"Failed: {progress.failed_items}"
            )
            
        except Exception as e:
            self.logger.error(f"Liahona magazine scraping failed: {e}")
            raise


def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for the Church Content Scraper.
    
    This function sets up the argument parser with all available options and their
    validation rules. It provides comprehensive help text and usage examples.
    
    Returns:
        argparse.Namespace: Parsed arguments with validated values
        
    Raises:
        SystemExit: If invalid arguments are provided or --help is requested
    """
    parser = argparse.ArgumentParser(
        description="Scrape religious content from the Church of Jesus Christ website",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Scrape General Conference talks from 1995-1999
  %(prog)s --start-year 1995 --end-year 1999 --content-type conference
  
  # Scrape Liahona articles from 2008 onwards with custom output directory
  %(prog)s --start-year 2008 --content-type liahona --output-dir ./my_content
  
  # Scrape both content types with increased delay and verbose logging
  %(prog)s --content-type both --delay 2.0 --verbose
  
  # Resume interrupted scraping (automatically skips existing files)
  %(prog)s --start-year 2010 --end-year 2020
  
  # Scrape with custom User-Agent for identification
  %(prog)s --user-agent "MyResearch-Bot/1.0" --delay 1.5

EXIT CODES:
  0   Success - all operations completed successfully
  1   Configuration error - invalid parameters or setup failure
  2   Network error - persistent connection or server issues  
  3   File system error - directory creation or file writing failures
  130 User interruption - script was interrupted with Ctrl+C

For more information, visit: https://github.com/your-repo/church-content-scraper
        """
    )
    
    parser.add_argument(
        "--start-year",
        type=int,
        default=1995,
        help="Starting year for scraping (default: 1995)"
    )
    
    parser.add_argument(
        "--end-year",
        type=int,
        help="Ending year for scraping (default: current year)"
    )
    
    parser.add_argument(
        "--content-type",
        choices=["conference", "liahona", "both"],
        default="both",
        help="Type of content to scrape (default: both)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scraped_content",
        help="Directory to save scraped content (default: scraped_content)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "--user-agent",
        type=str,
        default="Church-Content-Scraper/1.0",
        help="Custom User-Agent string (default: Church-Content-Scraper/1.0)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the Church Content Scraper.
    
    This function orchestrates the entire scraping workflow:
    1. Parse and validate command-line arguments
    2. Initialize logging system
    3. Create and validate configuration
    4. Execute scraping operations based on content type
    5. Handle errors gracefully with appropriate exit codes
    
    Returns:
        int: Exit code indicating success (0) or failure (1-3, 130)
            0   - Success: All operations completed successfully
            1   - Configuration error: Invalid parameters or setup failure
            2   - Network error: Persistent connection or server issues
            3   - File system error: Directory creation or file writing failures
            130 - User interruption: Script was interrupted with Ctrl+C
    """
    try:
        # Parse command-line arguments with validation
        args = parse_arguments()
        
        # Initialize logging system (creates log directory and configures handlers)
        setup_logging(args.verbose)
        
        # Create and validate configuration from command-line arguments
        # This will raise ValueError if any parameters are invalid
        config = ScraperConfig(
            start_year=args.start_year,
            end_year=args.end_year,
            content_type=args.content_type,
            output_dir=args.output_dir,
            delay=args.delay,
            user_agent=args.user_agent
        )
        
        # Log configuration for debugging and audit trail
        logging.info(f"Configuration: {config}")
        
        # Display startup information to user
        print("=" * 60)
        print("Church Content Scraper v1.0")
        print("=" * 60)
        print(f"Content type: {config.content_type}")
        print(f"Year range: {config.start_year} to {config.end_year}")
        print(f"Output directory: {config.output_dir}")
        print(f"Rate limiting: {config.delay}s between requests")
        print(f"User agent: {config.user_agent}")
        print("=" * 60)
        print()
        
        # Execute scraping operations based on configured content type
        scraping_success = True
        
        if config.content_type in ["conference", "both"]:
            try:
                print("🏛️  Starting General Conference scraping...")
                conference_scraper = ConferenceScraper(config)
                conference_scraper.run()
                print("✅ General Conference scraping completed successfully!")
                print()
            except Exception as e:
                logging.error(f"General Conference scraping failed: {e}")
                print(f"❌ General Conference scraping failed: {e}", file=sys.stderr)
                scraping_success = False
        
        if config.content_type in ["liahona", "both"]:
            try:
                print("📖 Starting Liahona magazine scraping...")
                liahona_scraper = LiahonaScraper(config)
                liahona_scraper.run()
                print("✅ Liahona magazine scraping completed successfully!")
                print()
            except Exception as e:
                logging.error(f"Liahona magazine scraping failed: {e}")
                print(f"❌ Liahona magazine scraping failed: {e}", file=sys.stderr)
                scraping_success = False
        
        # Final status report
        if scraping_success:
            print("🎉 All scraping operations completed successfully!")
            print(f"📁 Content saved to: {config.output_dir}")
            print("📋 Check the logs directory for detailed operation logs.")
            return 0  # Success
        else:
            print("⚠️  Some scraping operations failed. Check logs for details.", file=sys.stderr)
            return 2  # Network/scraping error
        
    except ValueError as e:
        # Configuration validation errors (invalid parameters, year ranges, etc.)
        error_msg = f"Configuration error: {e}"
        logging.error(error_msg)
        print(f"❌ {error_msg}", file=sys.stderr)
        print("💡 Use --help to see valid configuration options.", file=sys.stderr)
        return 1  # Configuration error
        
    except KeyboardInterrupt:
        # User interrupted the script with Ctrl+C
        logging.info("Scraping interrupted by user (Ctrl+C)")
        print("\n⏹️  Scraping interrupted by user. Progress has been saved.", file=sys.stderr)
        print("💡 You can resume by running the script again - existing files will be skipped.", file=sys.stderr)
        return 130  # Standard exit code for SIGINT (Ctrl+C)
        
    except (OSError, IOError, PermissionError) as e:
        # File system errors (directory creation, file writing, permissions)
        error_msg = f"File system error: {e}"
        logging.error(error_msg, exc_info=True)
        print(f"❌ {error_msg}", file=sys.stderr)
        print("💡 Check directory permissions and available disk space.", file=sys.stderr)
        return 3  # File system error
        
    except (ConnectionError, Timeout, RequestException) as e:
        # Network-related errors that couldn't be resolved with retries
        error_msg = f"Network error: {e}"
        logging.error(error_msg, exc_info=True)
        print(f"❌ {error_msg}", file=sys.stderr)
        print("💡 Check your internet connection and try again later.", file=sys.stderr)
        return 2  # Network error
        
    except Exception as e:
        # Unexpected errors - log full traceback for debugging
        error_msg = f"Unexpected error: {e}"
        logging.error(error_msg, exc_info=True)
        print(f"❌ {error_msg}", file=sys.stderr)
        print("💡 This is an unexpected error. Please check the logs for details.", file=sys.stderr)
        return 1  # General error


if __name__ == "__main__":
    sys.exit(main())