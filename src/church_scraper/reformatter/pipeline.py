"""
Content Reformatting Pipeline

Main pipeline for reformatting scraped content files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass

from .analyzer import ContentAnalyzer, ContentType
from .processor import TextProcessor
from .extractor import MetadataExtractor
from .formatter import ContentFormatter


class ProcessingResult(NamedTuple):
    """Result of processing a single file."""
    success: bool
    input_file: Path
    output_file: Optional[Path]
    content_type: ContentType
    message: str
    warnings: List[str]


@dataclass
class ReformatConfig:
    """Configuration for content reformatting."""
    input_dir: Path
    output_dir: Path
    overwrite_existing: bool = False
    skip_session_files: bool = True
    skip_invalid_content: bool = True
    create_backup: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)


class ContentReformatter:
    """Main content reformatting pipeline."""
    
    def __init__(self, config: ReformatConfig):
        self.config = config
        self.analyzer = ContentAnalyzer()
        self.processor = TextProcessor()
        self.extractor = MetadataExtractor()
        self.formatter = ContentFormatter()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Statistics tracking
        self.stats = {
            'processed': 0,
            'successful': 0,
            'skipped': 0,
            'failed': 0,
            'by_type': {}
        }

    def should_process_file(self, filepath: Path) -> bool:
        """Determine if a file should be processed."""
        if not filepath.suffix == '.txt':
            return False
        
        # Skip files ending with specific patterns if configured
        if self.config.skip_session_files:
            skip_patterns = [
                'session.txt',
                'sessionlangeng.txt'
            ]
            if any(filepath.name.lower().endswith(pattern) for pattern in skip_patterns):
                return False
        
        # Check if output already exists
        output_file = self.get_output_path(filepath)
        if output_file.exists() and not self.config.overwrite_existing:
            return False
        
        return True

    def get_output_path(self, input_file: Path) -> Path:
        """Generate output path for processed file."""
        # Preserve relative directory structure
        relative_path = input_file.relative_to(self.config.input_dir)
        
        # Change extension to indicate reformatted content
        output_name = relative_path.stem + '_formatted' + relative_path.suffix
        output_path = self.config.output_dir / relative_path.parent / output_name
        
        # Create parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return output_path

    def create_backup(self, filepath: Path) -> Optional[Path]:
        """Create backup of original file."""
        if not self.config.create_backup:
            return None
        
        backup_path = filepath.with_suffix(filepath.suffix + '.backup')
        try:
            backup_path.write_text(filepath.read_text(encoding='utf-8'), encoding='utf-8')
            return backup_path
        except Exception as e:
            self.logger.warning(f"Failed to create backup for {filepath}: {e}")
            return None

    def process_file(self, filepath: Path) -> ProcessingResult:
        """Process a single file through the reformatting pipeline."""
        self.logger.info(f"Processing {filepath}")
        
        try:
            # Read input file
            content = filepath.read_text(encoding='utf-8')
            
            # Analyze content type
            content_type = self.analyzer.detect_content_type(content, filepath)
            self.stats['by_type'][content_type.value] = self.stats['by_type'].get(content_type.value, 0) + 1
            
            # Skip non-talk content if configured
            if self.config.skip_invalid_content and content_type in [
                ContentType.SESSION_HEADER, 
                ContentType.TABLE_OF_CONTENTS, 
                ContentType.CONTENTS_PAGE,
                ContentType.INVALID
            ]:
                self.stats['skipped'] += 1
                return ProcessingResult(
                    success=False,
                    input_file=filepath,
                    output_file=None,
                    content_type=content_type,
                    message=f"Skipped {content_type.value} content",
                    warnings=[]
                )
            
            warnings = []
            
            # Extract clean content (handle TOC contamination)
            clean_content = self.analyzer.extract_clean_content(content)
            
            # Validate structure
            structure = self.analyzer.validate_structure(clean_content)
            if structure.issues:
                warnings.extend([f"Structure issue: {issue}" for issue in structure.issues])
            
            # Process text content
            processed_content, notes = self.processor.split_content_and_notes(clean_content)
            
            # Extract metadata from clean content
            metadata = self.extractor.extract_all_metadata(clean_content, filepath)
            
            # Clean timestamp from title if present
            if metadata.get('title'):
                metadata['title'] = self.processor.clean_timestamp_prefix(metadata['title'])
                
            metadata = self.extractor.infer_missing_metadata(metadata, filepath)
            
            # Validate essential metadata
            if not self.extractor.validate_metadata(metadata):
                warnings.append("Missing essential metadata")
            
            # Format content
            formatted_content = self.formatter.complete_formatting(metadata, processed_content, notes)
            
            # Validate output with content-type-specific thresholds
            if not self.formatter.validate_output(formatted_content, content_type):
                warnings.append("Output validation failed")
            
            # Create backup if requested
            if self.config.create_backup:
                self.create_backup(filepath)
            
            # Write output
            output_path = self.get_output_path(filepath)
            output_path.write_text(formatted_content, encoding='utf-8')
            
            self.stats['successful'] += 1
            
            return ProcessingResult(
                success=True,
                input_file=filepath,
                output_file=output_path,
                content_type=content_type,
                message="Successfully processed",
                warnings=warnings
            )
            
        except Exception as e:
            self.stats['failed'] += 1
            self.logger.error(f"Failed to process {filepath}: {e}")
            
            return ProcessingResult(
                success=False,
                input_file=filepath,
                output_file=None,
                content_type=ContentType.INVALID,
                message=f"Processing failed: {e}",
                warnings=[]
            )
        finally:
            self.stats['processed'] += 1

    def process_directory(self) -> List[ProcessingResult]:
        """Process all eligible files in the input directory."""
        self.logger.info(f"Starting batch processing of {self.config.input_dir}")
        
        # Find all .txt files
        txt_files = list(self.config.input_dir.rglob('*.txt'))
        eligible_files = [f for f in txt_files if self.should_process_file(f)]
        
        self.logger.info(f"Found {len(txt_files)} total files, {len(eligible_files)} eligible for processing")
        
        results = []
        
        for filepath in eligible_files:
            result = self.process_file(filepath)
            results.append(result)
            
            # Log progress periodically
            if len(results) % 50 == 0:
                self.logger.info(f"Processed {len(results)}/{len(eligible_files)} files")
        
        self.log_summary(results)
        return results

    def log_summary(self, results: List[ProcessingResult]):
        """Log processing summary."""
        self.logger.info("Processing Summary:")
        self.logger.info(f"  Total processed: {self.stats['processed']}")
        self.logger.info(f"  Successful: {self.stats['successful']}")
        self.logger.info(f"  Skipped: {self.stats['skipped']}")
        self.logger.info(f"  Failed: {self.stats['failed']}")
        
        if self.stats['by_type']:
            self.logger.info("  By content type:")
            for content_type, count in self.stats['by_type'].items():
                self.logger.info(f"    {content_type}: {count}")
        
        # Log files with warnings
        files_with_warnings = [r for r in results if r.warnings]
        if files_with_warnings:
            self.logger.warning(f"{len(files_with_warnings)} files had warnings:")
            for result in files_with_warnings[:10]:  # Show first 10
                self.logger.warning(f"  {result.input_file}: {', '.join(result.warnings)}")

    def process_single_file(self, filepath: Path) -> ProcessingResult:
        """Process a single specific file."""
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return self.process_file(filepath)