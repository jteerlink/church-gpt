#!/bin/bash

# Church-GPT Chat Interface - CLI Helper Script
# This script provides an easy way to start the Church-GPT chat interface

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
    print_color $BLUE "     Church-GPT Chat Interface"
    print_color $BLUE "=============================================="
    echo
}

# Function to show usage
show_usage() {
    print_color $YELLOW "Usage Examples:"
    echo "  ./start_chat.sh                           # Interactive mode"
    echo "  ./start_chat.sh --checkpoint /path/model  # Direct chat with model"
    echo "  ./start_chat.sh --api                     # Start API server"
    echo "  ./start_chat.sh --help                    # Show this help"
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
    
    # Check for required Python packages
    missing_packages=()
    
    if ! uv run python -c "import torch" &> /dev/null; then
        missing_packages+=("torch")
    fi
    
    if ! uv run python -c "import transformers" &> /dev/null; then
        missing_packages+=("transformers")
    fi
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_color $YELLOW "Missing packages: ${missing_packages[*]}"
        print_color $YELLOW "Installing with uv..."
        
        read -p "Install dependencies automatically? (y/n): " install_choice
        
        if [[ $install_choice =~ ^[Yy]$ ]]; then
            print_color $BLUE "Installing serving and API dependencies..."
            
            # Install serving and API dependencies
            if uv add torch transformers peft fastapi uvicorn; then
                print_color $GREEN "✓ Packages installed successfully"
                print_color $YELLOW "Note: PyTorch CPU version installed. For GPU support, run:"
                echo "  uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            else
                print_color $RED "Failed to install packages."
                exit 1
            fi
        else
            print_color $YELLOW "Please install dependencies manually:"
            echo "  uv add torch transformers peft fastapi uvicorn"
            exit 1
        fi
    fi
    
    print_color $GREEN "✓ Dependencies OK"
}

# Function to find model checkpoints
find_checkpoints() {
    local checkpoints=()
    
    # Common checkpoint locations
    local search_paths=(
        "./checkpoints"
        "./models"
        "./gemma3-7b-church"
        "."
    )
    
    for path in "${search_paths[@]}"; do
        if [ -d "$path" ]; then
            # Look for model files
            if ls "$path"/*.bin &> /dev/null || ls "$path"/pytorch_model.bin &> /dev/null || ls "$path"/model.safetensors &> /dev/null; then
                checkpoints+=("$path")
            fi
        fi
    done
    
    printf '%s\n' "${checkpoints[@]}"
}

# Function for interactive mode
interactive_mode() {
    print_header
    
    print_color $YELLOW "Select mode:"
    echo "1) Chat with model (interactive)"
    echo "2) Start API server"
    echo "3) Specify custom checkpoint path"
    echo "4) Exit"
    echo
    
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            print_color $GREEN "Starting interactive chat..."
            
            # Try to find checkpoints
            local checkpoints=($(find_checkpoints))
            
            if [ ${#checkpoints[@]} -eq 0 ]; then
                print_color $YELLOW "No model checkpoints found in common locations."
                read -p "Enter checkpoint path: " checkpoint_path
                if [ ! -d "$checkpoint_path" ]; then
                    print_color $RED "Checkpoint path does not exist: $checkpoint_path"
                    exit 1
                fi
            elif [ ${#checkpoints[@]} -eq 1 ]; then
                checkpoint_path="${checkpoints[0]}"
                print_color $GREEN "Using checkpoint: $checkpoint_path"
            else
                print_color $YELLOW "Multiple checkpoints found:"
                for i in "${!checkpoints[@]}"; do
                    echo "$((i+1))) ${checkpoints[$i]}"
                done
                read -p "Select checkpoint (1-${#checkpoints[@]}): " selection
                checkpoint_path="${checkpoints[$((selection-1))]}"
            fi
            
            uv run python serve.py --checkpoint "$checkpoint_path"
            ;;
        2)
            print_color $GREEN "Starting API server..."
            
            # Try to find checkpoints
            local checkpoints=($(find_checkpoints))
            
            if [ ${#checkpoints[@]} -eq 0 ]; then
                read -p "Enter checkpoint path: " checkpoint_path
            else
                checkpoint_path="${checkpoints[0]}"
                print_color $GREEN "Using checkpoint: $checkpoint_path"
            fi
            
            read -p "Port (default: 8000): " port
            port=${port:-8000}
            
            read -p "Host (default: 127.0.0.1): " host
            host=${host:-127.0.0.1}
            
            print_color $BLUE "Starting API server at http://$host:$port"
            uv run python serve.py --checkpoint "$checkpoint_path" --api --host "$host" --port "$port"
            ;;
        3)
            print_color $GREEN "Custom checkpoint configuration..."
            read -p "Checkpoint path: " checkpoint_path
            read -p "LoRA adapter path (optional): " lora_path
            read -p "Max new tokens (default: 256): " max_tokens
            read -p "Temperature (default: 0.7): " temperature
            
            cmd="uv run python serve.py --checkpoint '$checkpoint_path'"
            [ ! -z "$lora_path" ] && cmd="$cmd --lora_path '$lora_path'"
            [ ! -z "$max_tokens" ] && cmd="$cmd --max_new_tokens $max_tokens"
            [ ! -z "$temperature" ] && cmd="$cmd --temperature $temperature"
            
            eval $cmd
            ;;
        4)
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
        --checkpoint)
            if [ -z "$2" ]; then
                print_color $RED "Error: --checkpoint requires a path argument"
                exit 1
            fi
            print_header
            print_color $GREEN "Starting chat with checkpoint: $2"
            check_dependencies
            uv run python serve.py --checkpoint "$2"
            ;;
        --api)
            print_header
            print_color $GREEN "Starting API server..."
            check_dependencies
            
            # Try to find a checkpoint
            local checkpoints=($(find_checkpoints))
            if [ ${#checkpoints[@]} -eq 0 ]; then
                print_color $RED "No checkpoints found. Please specify with --checkpoint"
                exit 1
            fi
            
            checkpoint_path="${checkpoints[0]}"
            print_color $GREEN "Using checkpoint: $checkpoint_path"
            uv run python serve.py --checkpoint "$checkpoint_path" --api
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
}

# Run main function
main "$@"