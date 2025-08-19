#!/usr/bin/env python3
"""
Content Reformatting CLI Script

Command-line interface for reformatting scraped church content.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.church_scraper.reformatter import ContentReformatter, ReformatConfig


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Reformat scraped church content into standardized format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reformat all content in scraped_content directory
  python scripts/reformat_content.py scraped_content/ formatted_content/
  
  # Reformat with overwrite and backup
  python scripts/reformat_content.py scraped_content/ formatted_content/ --overwrite --backup
  
  # Reformat single file
  python scripts/reformat_content.py scraped_content/general-conference/1995-04/talk.txt formatted_content/ --single-file
  
  # Include session files and invalid content
  python scripts/reformat_content.py scraped_content/ formatted_content/ --include-sessions --include-invalid
        """
    )
    
    parser.add_argument(
        'input_path',
        type=Path,
        help='Input directory or file containing scraped content'
    )
    
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Output directory for formatted content'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing formatted files'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup of original files'
    )
    
    parser.add_argument(
        '--include-sessions',
        action='store_true',
        help='Include session header files (normally skipped)'
    )
    
    parser.add_argument(
        '--include-invalid',
        action='store_true',
        help='Include invalid/table-of-contents files (normally skipped)'
    )
    
    parser.add_argument(
        '--single-file',
        action='store_true',
        help='Process single file instead of directory'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually doing it'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input_path}")
    
    if args.single_file and not args.input_path.is_file():
        raise ValueError("--single-file specified but input is not a file")
    
    if not args.single_file and not args.input_path.is_dir():
        raise ValueError("Input path must be a directory when not using --single-file")


def main() -> int:
    """Main entry point."""
    try:
        args = parse_arguments()
        validate_arguments(args)
        setup_logging(args.verbose)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Starting content reformatting: {args.input_path} -> {args.output_dir}")
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No files will be modified")
        
        # Configure reformatter
        if args.single_file:
            # For single file, use parent directory as input_dir
            input_dir = args.input_path.parent
        else:
            input_dir = args.input_path
            
        config = ReformatConfig(
            input_dir=input_dir,
            output_dir=args.output_dir,
            overwrite_existing=args.overwrite,
            skip_session_files=not args.include_sessions,
            skip_invalid_content=not args.include_invalid,
            create_backup=args.backup,
            log_level='DEBUG' if args.verbose else 'INFO'
        )
        
        # Create reformatter
        reformatter = ContentReformatter(config)
        
        if args.dry_run:
            # In dry run mode, just analyze what would be processed
            if args.single_file:
                txt_files = [args.input_path] if args.input_path.suffix == '.txt' else []
            else:
                txt_files = list(args.input_path.rglob('*.txt'))
            
            eligible_files = [f for f in txt_files if reformatter.should_process_file(f)]
            
            logger.info(f"DRY RUN RESULTS:")
            logger.info(f"  Total .txt files found: {len(txt_files)}")
            logger.info(f"  Files that would be processed: {len(eligible_files)}")
            
            if args.verbose and eligible_files:
                logger.info("Files that would be processed:")
                for f in eligible_files[:10]:  # Show first 10
                    logger.info(f"  {f}")
                if len(eligible_files) > 10:
                    logger.info(f"  ... and {len(eligible_files) - 10} more")
            
            return 0
        
        # Process files
        if args.single_file:
            result = reformatter.process_single_file(args.input_path)
            
            if result.success:
                logger.info(f"Successfully processed: {result.input_file} -> {result.output_file}")
                if result.warnings:
                    logger.warning(f"Warnings: {', '.join(result.warnings)}")
            else:
                logger.error(f"Failed to process {result.input_file}: {result.message}")
                return 1
        else:
            results = reformatter.process_directory()
            
            # Check if any critical failures occurred
            failed_results = [r for r in results if not r.success and r.content_type.value != 'invalid']
            if failed_results:
                logger.error(f"{len(failed_results)} files failed to process")
                return 1
        
        logger.info("Content reformatting completed successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose if 'args' in locals() else False:
            logging.exception("Full exception details:")
        return 1


if __name__ == '__main__':
    sys.exit(main())