#!/usr/bin/env python3
"""
Test runner script for the church-gpt project.

This script sets up the proper Python path and runs tests.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the test suite with proper Python path setup."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.absolute()
    
    # Add project root to Python path
    env = os.environ.copy()
    pythonpath = str(project_root)
    if 'PYTHONPATH' in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env['PYTHONPATH'] = pythonpath
    
    # Change to project root
    os.chdir(project_root)
    
    # Run tests
    if len(sys.argv) > 1 and sys.argv[1] == '--unit':
        # Run only unit tests
        test_cmd = ['python', '-m', 'pytest', 'tests/unit/', '-v']
    elif len(sys.argv) > 1 and sys.argv[1] == '--integration':
        # Run only integration tests
        test_cmd = ['python', '-m', 'pytest', 'tests/integration/', '-v']
    else:
        # Run all tests
        test_cmd = ['python', '-m', 'pytest', 'tests/', '-v']
    
    print(f"🧪 Running tests from: {project_root}")
    print(f"📝 Command: {' '.join(test_cmd)}")
    print(f"🐍 PYTHONPATH: {pythonpath}")
    print()
    
    # Execute tests
    result = subprocess.run(test_cmd, env=env)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())