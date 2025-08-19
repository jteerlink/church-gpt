#!/usr/bin/env python3
"""
Main entry point for the church_scraper module.

This allows the module to be executed with:
python -m src.church_scraper
"""

from .core import main
import sys

if __name__ == "__main__":
    sys.exit(main())