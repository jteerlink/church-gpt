"""
Church Content Scraper

A comprehensive scraper for General Conference talks and Liahona articles
from the Church of Jesus Christ of Latter-day Saints website.
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

__version__ = "1.0.0"
__all__ = [
    "ScraperConfig",
    "ContentScraper", 
    "ConferenceScraper",
    "LiahonaScraper",
    "FileManager",
    "ProgressTracker",
    "RateLimiter",
    "setup_logging",
    "main",
]