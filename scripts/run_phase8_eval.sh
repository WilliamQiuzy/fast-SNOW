#!/bin/bash
# Phase 8 VLM4D Evaluation - One-Click Runner
#
# This script runs Phase 8 strict evaluation with fixed parameters
# for reproducibility and paper alignment.
#
# Requirements:
#   - GOOGLE_AI_API_KEY environment variable set
#   - VLM4D dataset at fast_snow/data/VLM4D-main/data/real_mc.json
#
# Usage:
#   bash scripts/run_phase8_eval.sh [OPTIONS]
#
# Options:
#   --quick-test    Run on 10 samples only (for testing)
#   --dry-run       Run without VLM calls (mock results)
#   --help          Show this help message

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
VLM4D_JSON="${PROJECT_ROOT}/fast_snow/data/VLM4D-main/data/real_mc.json"
RESULTS_DIR="${PROJECT_ROOT}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${RESULTS_DIR}/phase8_eval_${TIMESTAMP}.json"

# Flags
QUICK_TEST=false
DRY_RUN=false
PROCESS_VIDEOS=false
USE_MOCK_SCENES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --process-videos)
            PROCESS_VIDEOS=true
            shift
            ;;
        --use-mock-scenes)
            USE_MOCK_SCENES=true
            shift
            ;;
        --help)
            echo "Phase 8 VLM4D Evaluation - One-Click Runner"
            echo ""
            echo "Usage: bash scripts/run_phase8_eval.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick-test        Run on 10 samples only (for testing)"
            echo "  --dry-run           Run without VLM calls (mock results)"
            echo "  --process-videos    Process real videos (requires video files)"
            echo "  --use-mock-scenes   Use mock scenes instead of videos (default, faster)"
            echo "  --help              Show this help message"
            echo ""
            echo "Requirements:"
            echo "  - GOOGLE_AI_API_KEY environment variable set"
            echo "  - VLM4D dataset at fast_snow/data/VLM4D-main/data/real_mc.json"
            echo "  - (Optional) Video files if using --process-videos"
            echo ""
            echo "Output:"
            echo "  Results saved to results/phase8_eval_TIMESTAMP.json"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
echo ""
echo "======================================================================"
echo "  Phase 8 VLM4D Evaluation (Strict Reproduction)"
echo "======================================================================"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Model: google/gemma-3-4b-it (FIXED)"
echo "  Temperature: 0.0 (FIXED)"
echo "  Max tokens: 256 (FIXED)"
echo "  Backend: google_ai"
echo ""

# Check requirements
echo -e "${BLUE}Checking requirements...${NC}"

# Check API key
if [[ -z "${GOOGLE_AI_API_KEY}" ]]; then
    echo -e "${RED}✗ GOOGLE_AI_API_KEY not set${NC}"
    echo ""
    echo "Please set your Google AI Studio API key:"
    echo "  1. Get API key from https://aistudio.google.com/apikey"
    echo "  2. export GOOGLE_AI_API_KEY=your_api_key"
    echo ""
    exit 1
else
    echo -e "${GREEN}✓ GOOGLE_AI_API_KEY is set${NC}"
fi

# Check VLM4D dataset
if $DRY_RUN; then
    echo -e "${YELLOW}! DRY RUN MODE: VLM4D dataset check skipped${NC}"
elif [[ ! -f "$VLM4D_JSON" ]]; then
    echo -e "${YELLOW}! VLM4D dataset not found at: $VLM4D_JSON${NC}"
    echo ""
    echo "Will use mock samples for testing"
    echo ""
else
    echo -e "${GREEN}✓ VLM4D dataset found${NC}"
fi

# Create results directory
mkdir -p "$RESULTS_DIR"
echo -e "${GREEN}✓ Results directory ready: $RESULTS_DIR${NC}"

echo ""
echo "======================================================================"

# Build command
CMD="python ${SCRIPT_DIR}/run_phase8_vlm4d_eval.py"

# Add VLM4D JSON if exists
if [[ -f "$VLM4D_JSON" ]] && ! $DRY_RUN; then
    CMD="$CMD --vlm4d-json $VLM4D_JSON"
fi

# Add flags
if $QUICK_TEST; then
    CMD="$CMD --max-samples 10"
    echo -e "${YELLOW}Running quick test (10 samples)${NC}"
fi

if $DRY_RUN; then
    CMD="$CMD --dry-run"
    echo -e "${YELLOW}Running dry run (no VLM calls)${NC}"
fi

if $PROCESS_VIDEOS; then
    CMD="$CMD --process-videos"
    echo -e "${BLUE}Processing real videos with SNOW pipeline${NC}"
fi

if $USE_MOCK_SCENES || { ! $PROCESS_VIDEOS && ! $DRY_RUN; }; then
    CMD="$CMD --use-mock-scenes"
    echo -e "${YELLOW}Using mock scenes (no video processing)${NC}"
fi

# Add output path
CMD="$CMD --output $OUTPUT_FILE"

echo ""
echo -e "${BLUE}Running evaluation...${NC}"
echo "Command: $CMD"
echo ""
echo "======================================================================"
echo ""

# Run evaluation
if eval "$CMD"; then
    echo ""
    echo "======================================================================"
    echo -e "${GREEN}✓ Evaluation completed successfully!${NC}"
    echo ""
    echo "Results saved to:"
    echo "  $OUTPUT_FILE"
    echo ""

    # Show quick summary if jq is available
    if command -v jq &> /dev/null && [[ -f "$OUTPUT_FILE" ]]; then
        echo "Quick Summary:"
        echo "--------------"
        jq -r '.metrics | "Overall Accuracy: \(.overall_accuracy * 100 | round / 100)%\nTotal Samples: \(.total_samples)\nCorrect Samples: \(.correct_samples)"' "$OUTPUT_FILE" || true
        echo ""
    fi

    echo "To view detailed results:"
    echo "  cat $OUTPUT_FILE | jq ."
    echo ""
    echo "To compare with paper baseline:"
    echo "  jq '.metrics.category_metrics' $OUTPUT_FILE"
    echo ""
    echo "======================================================================"

    exit 0
else
    echo ""
    echo "======================================================================"
    echo -e "${RED}✗ Evaluation failed${NC}"
    echo ""
    echo "Please check the error messages above"
    echo ""
    echo "Common issues:"
    echo "  - API key expired or invalid"
    echo "  - Network connectivity issues"
    echo "  - VLM4D dataset path incorrect"
    echo ""
    echo "For help, see: docs/phases/PHASE8_EVAL_PROTOCOL.md"
    echo "======================================================================"

    exit 1
fi
