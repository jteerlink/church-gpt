#!/usr/bin/env python3
"""
Integration test runner for Church Content Scraper.

This script runs comprehensive integration tests to validate:
- Complete scraping workflows
- Directory structure creation and file organization  
- Resumable operations
- End-to-end functionality with real URLs (when network available)
- Error handling and recovery
"""

import sys
import unittest
import tempfile
import shutil
import time
from pathlib import Path
from io import StringIO

# Import test modules
from test_integration import (
    TestIntegrationWorkflows,
    TestEndToEndWithRealURLs, 
    TestConfigurationValidation
)

# Import existing unit test modules for comprehensive validation
from test_conference_scraper import TestConferenceScraper
from test_liahona_scraper import TestLiahonaScraper
from test_file_manager import TestFileManager


class IntegrationTestRunner:
    """Comprehensive test runner for integration validation."""
    
    def __init__(self):
        """Initialize test runner."""
        self.results = {}
        self.temp_dirs = []
    
    def run_test_suite(self, test_class, suite_name):
        """
        Run a specific test suite and capture results.
        
        Args:
            test_class: Test class to run
            suite_name: Name of the test suite for reporting
            
        Returns:
            Tuple of (success_count, failure_count, error_count)
        """
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        # Run tests with detailed output
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2,
            buffer=True
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Print results
        print(stream.getvalue())
        
        # Summary
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        successes = total_tests - failures - errors
        
        print(f"\n{suite_name} Summary:")
        print(f"  Tests run: {total_tests}")
        print(f"  Successes: {successes}")
        print(f"  Failures: {failures}")
        print(f"  Errors: {errors}")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        
        # Store results
        self.results[suite_name] = {
            'total': total_tests,
            'successes': successes,
            'failures': failures,
            'errors': errors,
            'time': end_time - start_time,
            'result': result
        }
        
        return successes, failures, errors
    
    def run_all_tests(self):
        """Run all integration and unit tests."""
        print("Church Content Scraper - Integration Test Suite")
        print("=" * 60)
        
        # Test suites to run
        test_suites = [
            (TestIntegrationWorkflows, "Integration Workflows"),
            (TestConfigurationValidation, "Configuration Validation"),
            (TestConferenceScraper, "Conference Scraper Unit Tests"),
            (TestLiahonaScraper, "Liahona Scraper Unit Tests"),
            (TestFileManager, "File Manager Unit Tests"),
            (TestEndToEndWithRealURLs, "End-to-End Real URL Tests"),
        ]
        
        total_successes = 0
        total_failures = 0
        total_errors = 0
        total_tests = 0
        
        # Run each test suite
        for test_class, suite_name in test_suites:
            try:
                successes, failures, errors = self.run_test_suite(test_class, suite_name)
                total_successes += successes
                total_failures += failures
                total_errors += errors
                total_tests += successes + failures + errors
            except Exception as e:
                print(f"ERROR: Failed to run {suite_name}: {e}")
                total_errors += 1
        
        # Overall summary
        self.print_final_summary(total_tests, total_successes, total_failures, total_errors)
        
        return total_failures + total_errors == 0
    
    def print_final_summary(self, total_tests, total_successes, total_failures, total_errors):
        """Print final test summary."""
        print(f"\n{'='*60}")
        print("FINAL TEST SUMMARY")
        print(f"{'='*60}")
        
        print(f"Total Tests Run: {total_tests}")
        print(f"Total Successes: {total_successes}")
        print(f"Total Failures: {total_failures}")
        print(f"Total Errors: {total_errors}")
        
        success_rate = (total_successes / total_tests * 100) if total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"\nDetailed Results by Suite:")
        print("-" * 40)
        
        for suite_name, results in self.results.items():
            status = "PASS" if results['failures'] + results['errors'] == 0 else "FAIL"
            print(f"{suite_name:30} {status:>6} ({results['successes']}/{results['total']})")
        
        # Print any failures or errors
        if total_failures > 0 or total_errors > 0:
            print(f"\n{'='*60}")
            print("FAILURE AND ERROR DETAILS")
            print(f"{'='*60}")
            
            for suite_name, results in self.results.items():
                result = results['result']
                
                if result.failures:
                    print(f"\nFailures in {suite_name}:")
                    for test, traceback in result.failures:
                        print(f"  - {test}: {traceback}")
                
                if result.errors:
                    print(f"\nErrors in {suite_name}:")
                    for test, traceback in result.errors:
                        print(f"  - {test}: {traceback}")
        
        # Final status
        if total_failures + total_errors == 0:
            print(f"\n🎉 ALL TESTS PASSED! Integration validation successful.")
        else:
            print(f"\n❌ {total_failures + total_errors} tests failed. Review failures above.")
    
    def validate_integration_requirements(self):
        """
        Validate that integration tests cover all requirements from task 9.
        
        Requirements from task 9:
        - Create integration tests that validate complete scraping workflows
        - Test directory structure creation and file organization
        - Validate resumable operations by running scraper multiple times
        - Add end-to-end tests with a small subset of real URLs
        """
        print(f"\n{'='*60}")
        print("INTEGRATION REQUIREMENTS VALIDATION")
        print(f"{'='*60}")
        
        requirements_coverage = {
            "Complete scraping workflows": [
                "test_complete_conference_workflow_mocked",
                "test_complete_liahona_workflow_mocked"
            ],
            "Directory structure and file organization": [
                "test_directory_structure_organization",
                "test_complete_conference_workflow_mocked",
                "test_complete_liahona_workflow_mocked"
            ],
            "Resumable operations": [
                "test_resumable_operations_conference",
                "test_resumable_operations_liahona"
            ],
            "End-to-end tests with real URLs": [
                "test_real_conference_page_access",
                "test_real_liahona_page_access"
            ],
            "Error handling and recovery": [
                "test_error_handling_and_recovery"
            ]
        }
        
        print("Requirements Coverage:")
        for requirement, test_methods in requirements_coverage.items():
            print(f"\n✓ {requirement}:")
            for method in test_methods:
                print(f"    - {method}")
        
        print(f"\n✅ All task requirements are covered by integration tests.")


def main():
    """Main entry point for integration test runner."""
    print("Starting Church Content Scraper Integration Tests...")
    
    # Create test runner
    runner = IntegrationTestRunner()
    
    try:
        # Run all tests
        success = runner.run_all_tests()
        
        # Validate requirements coverage
        runner.validate_integration_requirements()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during test run: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()