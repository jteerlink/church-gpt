#!/usr/bin/env python3
"""
Test the improved content cleaning on the problematic file.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.church_scraper import ContentScraper, ScraperConfig

def test_problematic_file():
    """Test the improved cleaning with the problematic file."""
    
    # Read the problematic file
    problem_file = "scraped_content/general-conference/2008-10/let-him-do-it-with-simplicitylangeng.txt"
    
    try:
        with open(problem_file, 'r', encoding='utf-8') as f:
            problem_content = f.read()
        
        print("=== ORIGINAL PROBLEMATIC CONTENT (first 500 chars) ===")
        print(problem_content[:500])
        print("\n" + "="*60 + "\n")
        
        # Create a test scraper to use the cleaning methods
        config = ScraperConfig()
        scraper = ContentScraper(config)
        
        # Test the full conference content cleaning
        cleaned_content = scraper._clean_conference_content(problem_content)
        print("=== CLEANED CONTENT ===")
        print(cleaned_content)
        
        # Show the difference in length
        print(f"\nOriginal length: {len(problem_content)} characters")
        print(f"Cleaned length: {len(cleaned_content)} characters")
        print(f"Reduction: {len(problem_content) - len(cleaned_content)} characters ({((len(problem_content) - len(cleaned_content)) / len(problem_content) * 100):.1f}%)")
        
        # Test that proper structure is present
        lines = cleaned_content.split('\n')
        if len(lines) > 0 and "Let Him Do It with Simplicity" in lines[0]:
            print("\n✅ Title properly extracted")
        else:
            print(f"\n❌ Title missing or incorrect: {lines[0] if lines else 'No content'}")
        
        if any("By Elder" in line for line in lines[:5]):
            print("✅ Author information present")
        else:
            print("❌ Author information missing")
            
        # Test encoding fixes
        if "Â" not in cleaned_content:
            print("✅ Non-breaking space encoding fixed")
        else:
            print("❌ Non-breaking space encoding still present")
            
        # Test table of contents removal
        if "ContentsSaturday Morning Session" not in cleaned_content:
            print("✅ Table of contents removed")
        else:
            print("❌ Table of contents still present")
            
        return cleaned_content
        
    except FileNotFoundError:
        print(f"File not found: {problem_file}")
        return None
    except Exception as e:
        print(f"Error testing: {e}")
        return None

if __name__ == "__main__":
    test_problematic_file()