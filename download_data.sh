#!/bin/bash

# Script to download all URLs from OLMo-mix file into /data directory
# Usage: ./download_data.sh [path_to_url_file] [start_index] [end_index] [--force]
# Example: ./download_data.sh file.txt 3000 4000
# Example with force redownload: ./download_data.sh file.txt 3000 4000 --force

set -e  # Exit on error

# Parse arguments
FORCE_REDOWNLOAD=false
for arg in "$@"; do
    if [ "$arg" == "--force" ] || [ "$arg" == "-f" ]; then
        FORCE_REDOWNLOAD=true
    fi
done

# Configuration
URL_FILE="${1:-/home/myh2014/code/ppt2/data_mixes/OLMo-mix-0625-150Bsample.txt}"
START_INDEX="${2:-1}"
END_INDEX="${3:-0}"  # 0 means download all
OUTPUT_DIR="/home/myh2014/code/ppt2/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if URL file exists
if [ ! -f "$URL_FILE" ]; then
    echo -e "${RED}Error: URL file not found: $URL_FILE${NC}"
    exit 1
fi

# Create output directory if it doesn't exist
echo -e "${GREEN}Creating output directory: $OUTPUT_DIR${NC}"
mkdir -p "$OUTPUT_DIR"

# Count total URLs
TOTAL_URLS=$(wc -l < "$URL_FILE")

# Set end index to total if not specified or if it exceeds total
if [ "$END_INDEX" -eq 0 ] || [ "$END_INDEX" -gt "$TOTAL_URLS" ]; then
    END_INDEX=$TOTAL_URLS
fi

# Validate indices
if [ "$START_INDEX" -lt 1 ]; then
    echo -e "${RED}Error: Start index must be >= 1${NC}"
    exit 1
fi

if [ "$START_INDEX" -gt "$END_INDEX" ]; then
    echo -e "${RED}Error: Start index ($START_INDEX) must be <= end index ($END_INDEX)${NC}"
    exit 1
fi

DOWNLOAD_COUNT=$((END_INDEX - START_INDEX + 1))
echo -e "${GREEN}Total URLs in file: $TOTAL_URLS${NC}"
echo -e "${GREEN}Downloading range: $START_INDEX to $END_INDEX ($DOWNLOAD_COUNT files)${NC}"
if [ "$FORCE_REDOWNLOAD" = true ]; then
    echo -e "${YELLOW}Force redownload mode: Will overwrite existing files${NC}"
fi

# Function to download a single file
download_file() {
    local url="$1"
    local count="$2"
    local total="$3"
    
    # Extract the path after the domain to preserve directory structure
    # e.g., http://olmo-data.org/preprocessed/dolma2-0625/... -> preprocessed/dolma2-0625/...
    local relative_path=$(echo "$url" | sed 's|http://olmo-data.org/||')
    local output_path="$OUTPUT_DIR/$relative_path"
    local output_dir=$(dirname "$output_path")
    
    # Create directory structure
    mkdir -p "$output_dir"
    
    # Download file if it doesn't exist or force redownload is enabled
    if [ -f "$output_path" ] && [ "$FORCE_REDOWNLOAD" = false ]; then
        echo -e "${YELLOW}[$count/$total] Skipping (already exists): $relative_path${NC}"
    else
        if [ -f "$output_path" ] && [ "$FORCE_REDOWNLOAD" = true ]; then
            echo -e "${YELLOW}[$count/$total] Force redownloading: $relative_path${NC}"
        else
            echo -e "${GREEN}[$count/$total] Downloading: $relative_path${NC}"
        fi
        
        if curl -f -L -o "$output_path" "$url" 2>/dev/null; then
            echo -e "${GREEN}[$count/$total] ✓ Completed: $relative_path${NC}"
        else
            echo -e "${RED}[$count/$total] ✗ Failed: $relative_path${NC}"
            rm -f "$output_path"  # Remove partial file
            return 1
        fi
    fi
    return 0
}

export -f download_file
export OUTPUT_DIR
export FORCE_REDOWNLOAD
export RED GREEN YELLOW NC

# Download files sequentially in the specified range
echo -e "${GREEN}Starting sequential download...${NC}"
count=$START_INDEX
sed -n "${START_INDEX},${END_INDEX}p" "$URL_FILE" | while IFS= read -r url; do
    download_file "$url" "$count" "$END_INDEX"
    count=$((count + 1))
done

echo -e "${GREEN}Download complete!${NC}"
echo -e "${GREEN}Files saved to: $OUTPUT_DIR${NC}"

# Show summary
echo -e "\n${GREEN}Summary:${NC}"
echo "Total files downloaded: $(find "$OUTPUT_DIR" -type f -name "*.npy" | wc -l)"
echo "Total size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
