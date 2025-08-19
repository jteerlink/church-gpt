# Design Document

## Overview

The Church Content Scraper is a Python script that extracts text content from the Church of Jesus Christ website, specifically targeting General Conference talks and Liahona magazine articles. The script will be organized as a modular, command-line application that can scrape content from configurable date ranges and save it to a structured local directory.

## Architecture

The application follows a modular architecture with clear separation of concerns:

```
church_scraper.py
├── Configuration Management
├── HTTP Session Management  
├── URL Generation
├── Content Scraping
├── File Management
└── Main Orchestration
```

### Core Components

1. **ScraperConfig**: Configuration management class
2. **ContentScraper**: Base scraping functionality
3. **ConferenceScraper**: General Conference specific scraping
4. **LiahonaScraper**: Liahona magazine specific scraping
5. **FileManager**: Local file operations and directory management

## Components and Interfaces

### ScraperConfig Class
```python
class ScraperConfig:
    def __init__(self, start_year=1995, end_year=None, output_dir="scraped_content", 
                 delay=1.0, user_agent="Church-Content-Scraper/1.0")
```
- Manages all configuration parameters
- Provides validation for year ranges and paths
- Handles default values and environment variable overrides

### ContentScraper Base Class
```python
class ContentScraper:
    def __init__(self, config: ScraperConfig)
    def setup_session(self) -> requests.Session
    def robust_get(self, url: str, **kwargs) -> requests.Response
    def extract_text(self, html: str) -> str
```
- Provides common HTTP session management with retry logic
- Implements exponential backoff for failed requests
- Handles text extraction from HTML using BeautifulSoup

### ConferenceScraper Class
```python
class ConferenceScraper(ContentScraper):
    def get_conference_urls(self) -> List[str]
    def scrape_conference_page(self, conf_url: str) -> List[str]
    def scrape_talk(self, talk_url: str) -> str
    def run(self) -> None
```
- Generates conference URLs for April and October sessions
- Extracts individual talk URLs from conference index pages
- Downloads and processes individual talk content

### LiahonaScraper Class  
```python
class LiahonaScraper(ContentScraper):
    def get_monthly_urls(self) -> List[Tuple[int, int, str]]
    def scrape_monthly_page(self, month_url: str) -> List[str]
    def scrape_article(self, article_url: str) -> str
    def run(self) -> None
```
- Generates monthly Liahona URLs (excluding conference months)
- Extracts article URLs from monthly index pages
- Downloads and processes individual article content

### FileManager Class
```python
class FileManager:
    def __init__(self, base_dir: str)
    def create_directory_structure(self, year: int, month: int = None) -> str
    def save_content(self, content: str, filepath: str) -> None
    def file_exists(self, filepath: str) -> bool
```
- Manages local directory creation and organization
- Handles file writing with proper encoding
- Provides utilities for checking existing files to avoid re-downloading

## Data Models

### URL Structure
- Conference URLs: `https://www.churchofjesuschrist.org/study/general-conference/{year}/{month:02d}?lang=eng`
- Talk URLs: `https://www.churchofjesuschrist.org/study/general-conference/{year}/{month:02d}/{slug}?lang=eng`
- Liahona URLs: `https://www.churchofjesuschrist.org/study/liahona/{year}/{month:02d}?lang=eng`
- Article URLs: `https://www.churchofjesuschrist.org/study/liahona/{year}/{month:02d}/{slug}?lang=eng`

### Directory Structure
```
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
```

### Content Processing
- Extract text using BeautifulSoup's `get_text()` method
- Preserve paragraph structure with newline separators
- Save content with UTF-8 encoding
- Generate filenames from URL slugs

## Error Handling

### HTTP Error Handling
- Implement retry logic with exponential backoff (up to 10 attempts)
- Handle specific exceptions: `SSLError`, `ReadTimeout`, `ConnectionError`
- Use appropriate timeouts for different request types (HEAD vs GET)
- Log all retry attempts and final failures

### File System Error Handling
- Create directories recursively with proper permissions
- Handle file writing errors with informative messages
- Validate file paths and names for cross-platform compatibility
- Skip existing files to allow resumable operations

### Network Resilience
- Use persistent HTTP sessions for connection pooling
- Implement proper User-Agent headers to avoid blocking
- Add configurable delays between requests to respect server resources
- Handle rate limiting with appropriate backoff strategies

## Testing Strategy

### Unit Tests
- Test URL generation for different year ranges and content types
- Test HTML parsing and text extraction with sample content
- Test file management operations (create, write, check existence)
- Test configuration validation and error handling

### Integration Tests
- Test complete scraping workflow with mock HTTP responses
- Test directory structure creation and file organization
- Test error recovery and retry mechanisms
- Test resumable operations (skipping existing files)

### End-to-End Tests
- Test against a small subset of real URLs (with appropriate delays)
- Validate output file structure and content quality
- Test command-line interface and configuration options
- Performance testing with rate limiting validation

### Test Data Management
- Create sample HTML files representing different page structures
- Mock HTTP responses for reliable testing
- Use temporary directories for file system tests
- Implement cleanup procedures for test artifacts