#!/bin/bash

# Church Content Scraper - CLI Helper Script
# This script provides an interactive interface for running the church content scraper

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Function to print header
print_header() {
    echo
    print_color $BLUE "=============================================="
    print_color $BLUE "  Church Content Scraper - CLI Interface"
    print_color $BLUE "=============================================="
    echo
}

# Function to show usage
show_usage() {
    print_color $YELLOW "Usage Examples:"
    echo "  ./start_scraper.sh                    # Interactive mode"
    echo "  ./start_scraper.sh --quick-conference # Quick conference scrape (2020+)"
    echo "  ./start_scraper.sh --quick-liahona    # Quick Liahona scrape (2020+)"
    echo "  ./start_scraper.sh --help             # Show this help"
    echo
}

# Function to check and install uv
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_color $YELLOW "uv is not installed. Installing uv..."
        if command -v curl &> /dev/null; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
        elif command -v brew &> /dev/null; then
            brew install uv
        else
            print_color $RED "Error: Cannot install uv. Please install curl or homebrew first."
            print_color $BLUE "Visit https://docs.astral.sh/uv/getting-started/installation/ for manual installation."
            exit 1
        fi
        
        # Verify installation
        if ! command -v uv &> /dev/null; then
            print_color $RED "Error: uv installation failed."
            exit 1
        fi
        
        print_color $GREEN "✓ uv installed successfully"
    fi
}

# Function to check dependencies
check_dependencies() {
    print_color $BLUE "Checking dependencies..."
    
    # Check and install uv if needed
    check_uv
    
    # Check if we're in a uv project or need to create one
    if [ ! -f "pyproject.toml" ] && [ ! -f ".python-version" ]; then
        print_color $YELLOW "Initializing uv project..."
        uv init --no-readme --no-workspace
        print_color $GREEN "✓ uv project initialized"
    fi
    
    # Check if dependencies are installed
    if ! uv run python -c "import requests, bs4" &> /dev/null; then
        print_color $YELLOW "Installing required Python packages with uv..."
        
        read -p "Install dependencies automatically? (y/n): " install_choice
        
        if [[ $install_choice =~ ^[Yy]$ ]]; then
            print_color $BLUE "Installing scraping dependencies..."
            if uv add requests beautifulsoup4; then
                print_color $GREEN "✓ Packages installed successfully"
            else
                print_color $RED "Failed to install packages."
                exit 1
            fi
        else
            print_color $YELLOW "Please install dependencies manually:"
            echo "  uv add requests beautifulsoup4"
            exit 1
        fi
    fi
    
    print_color $GREEN "✓ Dependencies OK"
}

# Function for interactive mode
interactive_mode() {
    print_header
    
    print_color $YELLOW "Select content type to scrape:"
    echo "1) General Conference talks only"
    echo "2) Liahona magazine articles only"
    echo "3) Both Conference and Liahona"
    echo "4) Custom configuration"
    echo "5) Exit"
    echo
    
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            print_color $GREEN "Scraping General Conference talks..."
            read -p "Start year (default: 1995): " start_year
            start_year=${start_year:-1995}
            uv run python church_scraper.py --content-type conference --start-year $start_year --verbose
            ;;
        2)
            print_color $GREEN "Scraping Liahona articles..."
            read -p "Start year (default: 2008): " start_year
            start_year=${start_year:-2008}
            uv run python church_scraper.py --content-type liahona --start-year $start_year --verbose
            ;;
        3)
            print_color $GREEN "Scraping both content types..."
            read -p "Start year (default: 2020): " start_year
            start_year=${start_year:-2020}
            uv run python church_scraper.py --content-type both --start-year $start_year --verbose
            ;;
        4)
            print_color $GREEN "Custom configuration..."
            read -p "Start year: " start_year
            read -p "End year (default: current): " end_year
            read -p "Content type (conference/liahona/both): " content_type
            read -p "Delay between requests (default: 1.0): " delay
            read -p "Output directory (default: scraped_content): " output_dir
            
            cmd="uv run python church_scraper.py --verbose"
            [ ! -z "$start_year" ] && cmd="$cmd --start-year $start_year"
            [ ! -z "$end_year" ] && cmd="$cmd --end-year $end_year"
            [ ! -z "$content_type" ] && cmd="$cmd --content-type $content_type"
            [ ! -z "$delay" ] && cmd="$cmd --delay $delay"
            [ ! -z "$output_dir" ] && cmd="$cmd --output-dir $output_dir"
            
            eval $cmd
            ;;
        5)
            print_color $YELLOW "Exiting..."
            exit 0
            ;;
        *)
            print_color $RED "Invalid choice. Please try again."
            interactive_mode
            ;;
    esac
}

# Main script logic
main() {
    # Handle command line arguments
    case "${1:-}" in
        --help|-h)
            print_header
            show_usage
            exit 0
            ;;
        --quick-conference)
            print_header
            print_color $GREEN "Quick Conference scrape (2020+)..."
            check_dependencies
            uv run python church_scraper.py --content-type conference --start-year 2020 --verbose
            ;;
        --quick-liahona)
            print_header
            print_color $GREEN "Quick Liahona scrape (2020+)..."
            check_dependencies
            uv run python church_scraper.py --content-type liahona --start-year 2020 --verbose
            ;;
        "")
            check_dependencies
            interactive_mode
            ;;
        *)
            print_color $RED "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
    
    print_color $GREEN "✓ Scraping completed successfully!"
    print_color $BLUE "Check the 'scraped_content' directory for your files."
}

# Run main function
main "$@"