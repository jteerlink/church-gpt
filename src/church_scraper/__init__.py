"""
Church Content Scraper

A comprehensive scraper for General Conference talks and Liahona articles
from the Church of Jesus Christ of Latter-day Saints website.

Includes content reformatting capabilities for standardizing scraped text.
"""

from .core import (
    ScraperConfig,
    ContentScraper,
    ConferenceScraper,
    LiahonaScraper,
    FileManager,
    ProgressTracker,
    RateLimiter,
    setup_logging,
    main,
)

from .reformatter import (
    ContentAnalyzer,
    TextProcessor,
    MetadataExtractor,
    ContentFormatter,
    ContentReformatter,
    ReformatConfig,
)

__version__ = "1.1.0"
__all__ = [
    # Core scraper components
    "ScraperConfig",
    "ContentScraper", 
    "ConferenceScraper",
    "LiahonaScraper",
    "FileManager",
    "ProgressTracker",
    "RateLimiter",
    "setup_logging",
    "main",
    # Reformatter components
    "ContentAnalyzer",
    "TextProcessor",
    "MetadataExtractor", 
    "ContentFormatter",
    "ContentReformatter",
    "ReformatConfig",
]