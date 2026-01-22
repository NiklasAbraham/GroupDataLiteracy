#!/bin/bash

# Script to run all neighbourhood analysis files sequentially in the conda environment
# 
# Usage:
#   ./run_neighbourhood_tests.sh                    # Stop on first error
#   ./run_neighbourhood_tests.sh --continue-on-error # Continue even if scripts fail
#
# Note: Scripts require data files in data/data_final/. If data is missing,
#       scripts will fail with FileNotFoundError (this is expected behavior).

set -e  # Exit on error (can be disabled with --continue-on-error flag)

# Configuration
CONDA_ENV="dataLiteracy"
SCRIPT_DIR="src/aab_analysis/neighbourhood"
LOG_DIR="logs"
STOP_ON_ERROR=true

# Parse command line arguments
if [[ "$1" == "--continue-on-error" ]]; then
    STOP_ON_ERROR=false
    set +e  # Don't exit on error
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to run a script
run_script() {
    local script_name=$1
    local script_path="$SCRIPT_DIR/$script_name"
    local log_file="$LOG_DIR/${script_name%.py}.log"
    
    print_status "Running $script_name..."
    echo "========================================" >> "$log_file"
    echo "Script: $script_name" >> "$log_file"
    echo "Started at: $(date)" >> "$log_file"
    echo "Working directory: $(pwd)" >> "$log_file"
    echo "Script path: $script_path" >> "$log_file"
    echo "========================================" >> "$log_file"
    
    # Run from project root with PYTHONPATH set to current directory
    # Use absolute path for script to avoid any path issues
    local abs_script_path="$(pwd)/$script_path"
    if PYTHONPATH="$(pwd):${PYTHONPATH:-}" conda run -n "$CONDA_ENV" python "$abs_script_path" >> "$log_file" 2>&1; then
        print_success "$script_name completed successfully"
        echo "Completed at: $(date)" >> "$log_file"
        echo "Status: SUCCESS" >> "$log_file"
        return 0
    else
        print_error "$script_name failed (check $log_file for details)"
        echo "Completed at: $(date)" >> "$log_file"
        echo "Status: FAILED" >> "$log_file"
        # Show last few lines of error for quick debugging
        echo ""
        print_warning "Last 10 lines of error log:"
        tail -10 "$log_file" | sed 's/^/  /'
        echo ""
        return 1
    fi
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    exit 1
fi

# Get the script's directory and change to project root
SCRIPT_LOCATION="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_LOCATION" || exit 1

print_status "Changed to project root: $(pwd)"

# Initialize conda (required for conda run)
# Suppress conda warnings/errors during initialization
eval "$(conda shell.bash hook 2>/dev/null)" || {
    print_error "Failed to initialize conda. Make sure conda is installed and in PATH."
    exit 1
}

# Test if we can use conda run (this will fail if environment doesn't exist)
print_status "Testing conda environment access..."
if ! conda run -n "$CONDA_ENV" python -c "import sys; print('OK')" 2>/dev/null; then
    print_error "Cannot access conda environment '$CONDA_ENV'"
    print_status "Make sure the environment exists. Create it with: conda env create -f environment.yml"
    exit 1
fi

print_status "Using conda environment: $CONDA_ENV"
print_status "Log directory: $LOG_DIR"
print_status "Script directory: $SCRIPT_DIR"
print_status "Project root: $(pwd)"
echo ""

# List of scripts to run in order
# Note: 001 and 002 are module files (not runnable scripts), so we skip them
scripts=(
    "003_find_neighbors.py"
    "004_find_dissimilar.py"
    "005_gaussian_fit.py"
    "006_keyword_group_analysis.py"
    "007_neighbor_distribution.py"
    "008_cosine_statistics.py"
    "009_extract_high_distance_movies.py"
)

# Track results
total_scripts=${#scripts[@]}
successful=0
failed=0
failed_scripts=()

print_status "Starting test run for $total_scripts scripts..."
print_status "Note: 001_gaussian_analysis.py and 002_neighbor_utils.py are module files (not runnable scripts) and are skipped"
echo ""

# Run each script
for script in "${scripts[@]}"; do
    script_path="$SCRIPT_DIR/$script"
    
    # Check if script exists
    if [[ ! -f "$script_path" ]]; then
        print_warning "Script $script not found, skipping..."
        ((failed++))
        failed_scripts+=("$script (not found)")
        continue
    fi
    
    # Run the script
    if run_script "$script"; then
        ((successful++))
    else
        ((failed++))
        failed_scripts+=("$script")
        
        if [[ "$STOP_ON_ERROR" == true ]]; then
            print_error "Stopping due to error (use --continue-on-error to continue)"
            break
        fi
    fi
    
    echo ""
done

# Print summary
echo ""
echo "========================================"
print_status "Test Run Summary"
echo "========================================"
echo "Total scripts: $total_scripts"
print_success "Successful: $successful"
if [[ $failed -gt 0 ]]; then
    print_error "Failed: $failed"
    echo ""
    print_error "Failed scripts:"
    for failed_script in "${failed_scripts[@]}"; do
        echo "  - $failed_script"
    done
else
    print_success "All scripts completed successfully!"
fi
echo ""
print_status "Log files are available in: $LOG_DIR"
echo "========================================"

# Exit with error code if any failed
if [[ $failed -gt 0 ]]; then
    exit 1
else
    exit 0
fi
