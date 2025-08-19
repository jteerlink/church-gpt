#!/usr/bin/env python3
"""
Unit tests for FileManager class in church_scraper.py

Tests file operations, directory creation, and error handling using temporary directories.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from church_scraper import FileManager


class TestFileManager(unittest.TestCase):
    """Test cases for FileManager class."""
    
    def setUp(self):
        """Set up test fixtures with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.file_manager = FileManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_creates_base_directory(self):
        """Test that FileManager creates base directory on initialization."""
        # Test with new directory
        new_temp_dir = os.path.join(self.temp_dir, "new_base")
        fm = FileManager(new_temp_dir)
        
        self.assertTrue(Path(new_temp_dir).exists())
        self.assertTrue(Path(new_temp_dir).is_dir())
        self.assertEqual(str(fm.base_dir), str(Path(new_temp_dir).resolve()))
    
    def test_create_directory_structure_with_month(self):
        """Test creating directory structure with year and month."""
        # Test general conference directory
        dir_path = self.file_manager.create_directory_structure(
            year=2023, 
            month=4, 
            content_type="general-conference"
        )
        
        expected_path = Path(self.temp_dir) / "general-conference" / "2023-04"
        self.assertEqual(dir_path, str(expected_path.resolve()))
        self.assertTrue(expected_path.exists())
        self.assertTrue(expected_path.is_dir())
        
        # Test liahona directory
        dir_path = self.file_manager.create_directory_structure(
            year=2023, 
            month=1, 
            content_type="liahona"
        )
        
        expected_path = Path(self.temp_dir) / "liahona" / "2023-01"
        self.assertEqual(dir_path, str(expected_path.resolve()))
        self.assertTrue(expected_path.exists())
        self.assertTrue(expected_path.is_dir())
    
    def test_create_directory_structure_without_month(self):
        """Test creating directory structure with only year."""
        dir_path = self.file_manager.create_directory_structure(
            year=2023, 
            content_type="general-conference"
        )
        
        expected_path = Path(self.temp_dir) / "general-conference" / "2023"
        self.assertEqual(dir_path, str(expected_path.resolve()))
        self.assertTrue(expected_path.exists())
        self.assertTrue(expected_path.is_dir())
    
    def test_create_directory_structure_validation(self):
        """Test validation of directory structure parameters."""
        # Test invalid year
        with self.assertRaises(ValueError) as cm:
            self.file_manager.create_directory_structure(year=1800)
        self.assertIn("Year must be an integer between 1900 and 2100", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            self.file_manager.create_directory_structure(year=2200)
        self.assertIn("Year must be an integer between 1900 and 2100", str(cm.exception))
        
        # Test invalid month
        with self.assertRaises(ValueError) as cm:
            self.file_manager.create_directory_structure(year=2023, month=0)
        self.assertIn("Month must be an integer between 1 and 12", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            self.file_manager.create_directory_structure(year=2023, month=13)
        self.assertIn("Month must be an integer between 1 and 12", str(cm.exception))
        
        # Test invalid content type
        with self.assertRaises(ValueError) as cm:
            self.file_manager.create_directory_structure(year=2023, content_type="invalid")
        self.assertIn("content_type must be one of", str(cm.exception))
    
    def test_save_content(self):
        """Test saving content to files."""
        # Create directory first
        dir_path = self.file_manager.create_directory_structure(2023, 4, "general-conference")
        filepath = os.path.join(dir_path, "test-content.txt")
        
        test_content = "This is test content\nwith multiple lines\nand unicode: café"
        
        # Save content
        self.file_manager.save_content(test_content, filepath)
        
        # Verify file exists and content is correct
        self.assertTrue(Path(filepath).exists())
        
        with open(filepath, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        
        self.assertEqual(saved_content, test_content)
    
    def test_save_content_creates_parent_directories(self):
        """Test that save_content creates parent directories if they don't exist."""
        # Use a nested path that doesn't exist yet
        nested_path = os.path.join(self.temp_dir, "deep", "nested", "path", "file.txt")
        test_content = "Test content for nested path"
        
        # Save content - should create all parent directories
        self.file_manager.save_content(test_content, nested_path)
        
        # Verify file exists and content is correct
        self.assertTrue(Path(nested_path).exists())
        
        with open(nested_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        
        self.assertEqual(saved_content, test_content)
    
    def test_save_content_validation(self):
        """Test validation of save_content parameters."""
        # Test empty content
        with self.assertRaises(ValueError) as cm:
            self.file_manager.save_content("", "test.txt")
        self.assertIn("Content must be a non-empty string", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            self.file_manager.save_content("   ", "test.txt")
        self.assertIn("Content must be a non-empty string", str(cm.exception))
        
        # Test empty filepath
        with self.assertRaises(ValueError) as cm:
            self.file_manager.save_content("content", "")
        self.assertIn("Filepath must be a non-empty string", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            self.file_manager.save_content("content", "   ")
        self.assertIn("Filepath must be a non-empty string", str(cm.exception))
    
    def test_file_exists(self):
        """Test file existence checking."""
        # Create a test file
        dir_path = self.file_manager.create_directory_structure(2023, 4, "general-conference")
        filepath = os.path.join(dir_path, "existing-file.txt")
        
        # File doesn't exist yet
        self.assertFalse(self.file_manager.file_exists(filepath))
        
        # Create the file
        self.file_manager.save_content("test content", filepath)
        
        # File should exist now
        self.assertTrue(self.file_manager.file_exists(filepath))
        
        # Test non-existent file
        non_existent = os.path.join(dir_path, "non-existent.txt")
        self.assertFalse(self.file_manager.file_exists(non_existent))
    
    def test_file_exists_with_invalid_paths(self):
        """Test file_exists with invalid or empty paths."""
        # Test empty string
        self.assertFalse(self.file_manager.file_exists(""))
        
        # Test whitespace only
        self.assertFalse(self.file_manager.file_exists("   "))
        
        # Test invalid path characters (should handle gracefully)
        self.assertFalse(self.file_manager.file_exists("\x00invalid"))
    
    def test_get_content_filepath(self):
        """Test generating content filepaths."""
        # Test general conference filepath
        filepath = self.file_manager.get_content_filepath(
            year=2023, 
            month=4, 
            slug="test-talk", 
            content_type="general-conference"
        )
        
        expected_path = str((Path(self.temp_dir) / "general-conference" / "2023-04" / "test-talk.txt").resolve())
        self.assertEqual(filepath, expected_path)
        
        # Verify directory was created
        self.assertTrue((Path(self.temp_dir) / "general-conference" / "2023-04").exists())
        
        # Test liahona filepath
        filepath = self.file_manager.get_content_filepath(
            year=2023, 
            month=1, 
            slug="test-article", 
            content_type="liahona"
        )
        
        expected_path = str((Path(self.temp_dir) / "liahona" / "2023-01" / "test-article.txt").resolve())
        self.assertEqual(filepath, expected_path)
    
    def test_get_content_filepath_slug_cleaning(self):
        """Test that get_content_filepath cleans slugs properly."""
        # Test slug with special characters
        filepath = self.file_manager.get_content_filepath(
            year=2023, 
            month=4, 
            slug="test/talk*with?special<chars>", 
            content_type="general-conference"
        )
        
        # Should clean special characters
        self.assertIn("testtalkwithspecialchars.txt", filepath)
        
        # Test slug that already has .txt extension
        filepath = self.file_manager.get_content_filepath(
            year=2023, 
            month=4, 
            slug="test-talk.txt", 
            content_type="general-conference"
        )
        
        # Should not double the extension
        self.assertTrue(filepath.endswith("test-talk.txt"))
        self.assertFalse(filepath.endswith("test-talk.txt.txt"))
        
        # Test empty slug - should raise ValueError
        with self.assertRaises(ValueError) as cm:
            self.file_manager.get_content_filepath(
                year=2023, 
                month=4, 
                slug="", 
                content_type="general-conference"
            )
        self.assertIn("Slug must be a non-empty string", str(cm.exception))
    
    def test_get_content_filepath_validation(self):
        """Test validation of get_content_filepath parameters."""
        # Test invalid year
        with self.assertRaises(ValueError):
            self.file_manager.get_content_filepath(1800, 4, "slug")
        
        # Test invalid month
        with self.assertRaises(ValueError):
            self.file_manager.get_content_filepath(2023, 0, "slug")
        
        # Test invalid content type
        with self.assertRaises(ValueError):
            self.file_manager.get_content_filepath(2023, 4, "slug", "invalid")
    
    def test_list_existing_files(self):
        """Test listing existing files in directories."""
        # Create some test files
        dir_path = self.file_manager.create_directory_structure(2023, 4, "general-conference")
        
        # Initially no files
        files = self.file_manager.list_existing_files(2023, 4, "general-conference")
        self.assertEqual(files, [])
        
        # Create some files
        test_files = ["talk1.txt", "talk2.txt", "talk3.txt"]
        for filename in test_files:
            filepath = os.path.join(dir_path, filename)
            self.file_manager.save_content(f"Content for {filename}", filepath)
        
        # List files
        files = self.file_manager.list_existing_files(2023, 4, "general-conference")
        
        # Should return all .txt files
        self.assertEqual(len(files), 3)
        for filename in test_files:
            self.assertIn(filename, files)
        
        # Create a non-.txt file (should be ignored)
        non_txt_path = os.path.join(dir_path, "not-text.html")
        with open(non_txt_path, 'w') as f:
            f.write("HTML content")
        
        # List files again - should still only return .txt files
        files = self.file_manager.list_existing_files(2023, 4, "general-conference")
        self.assertEqual(len(files), 3)
        self.assertNotIn("not-text.html", files)
    
    def test_list_existing_files_without_month(self):
        """Test listing files in year-only directories."""
        # Create directory without month
        dir_path = self.file_manager.create_directory_structure(2023, content_type="general-conference")
        
        # Create test file
        filepath = os.path.join(dir_path, "yearly-content.txt")
        self.file_manager.save_content("Yearly content", filepath)
        
        # List files
        files = self.file_manager.list_existing_files(2023, content_type="general-conference")
        
        self.assertEqual(len(files), 1)
        self.assertIn("yearly-content.txt", files)
    
    def test_list_existing_files_nonexistent_directory(self):
        """Test listing files in non-existent directories."""
        # Try to list files in directory that doesn't exist
        files = self.file_manager.list_existing_files(2099, 12, "general-conference")
        
        # Should return empty list, not raise error
        self.assertEqual(files, [])
    
    @patch('church_scraper.Path.mkdir')
    def test_create_directory_structure_os_error(self, mock_mkdir):
        """Test handling of OS errors during directory creation."""
        # Mock mkdir to raise OSError
        mock_mkdir.side_effect = OSError("Permission denied")
        
        with self.assertRaises(OSError) as cm:
            self.file_manager.create_directory_structure(2023, 4, "general-conference")
        
        self.assertIn("Failed to create directory structure", str(cm.exception))
    
    @patch('builtins.open')
    def test_save_content_os_error(self, mock_open):
        """Test handling of OS errors during file writing."""
        # Mock open to raise OSError
        mock_open.side_effect = OSError("Disk full")
        
        with self.assertRaises(OSError) as cm:
            self.file_manager.save_content("content", "test.txt")
        
        self.assertIn("Failed to save content to file", str(cm.exception))


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()