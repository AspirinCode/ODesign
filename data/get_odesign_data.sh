#!/bin/bash

#######################################################################
# ODesign Data Download Script
# ---------------------------------------------------------------------
# This script downloads required data files for running ODesign.
# 
# Usage:
#     bash get_odesign_data.sh [data_root_dir] [inference_only]
#
# Parameters:
#     1. data_root_dir: User-specified data storage root directory 
#                       (default: ./data)
#     2. inference_only: Whether to only download inference data 
#                       (default: true)
#######################################################################

# Set default arguments
DEFAULT_DATA_ROOT_DIR="./data"
DEFAULT_INFERENCE_ONLY="true"

# Parse command line arguments
data_root_dir="${1:-$DEFAULT_DATA_ROOT_DIR}"
inference_only="${2:-$DEFAULT_INFERENCE_ONLY}"

# Inference required data files
INFERENCE_FILES=(
    "https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif"
    "https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl"
)

# Training required data files
TRAINING_FILES=(
    # Training files will be added here in the future
)

#######################################################################
# Config Summary
#######################################################################

echo "-----------------------------------------------------------"
echo "🚀 Start ODesign Data Download"
echo "-----------------------------------------------------------"
echo "Data Root Directory: $data_root_dir"
echo "Inference Only     : $inference_only"
echo "-----------------------------------------------------------"
echo ""

# Create data directory if it doesn't exist
mkdir -p "$data_root_dir"

# Function to download file with error handling
download_file() {
    local file_name=$1
    local file_url=$2
    local output_path="$data_root_dir/$file_name"
    
    echo "📥 Downloading: $file_name"
    echo "   From: $file_url"
    echo "   To: $output_path"
    
    # Check if file already exists
    if [[ -f "$output_path" ]]; then
        echo "   ⚠️  File already exists, skipping download."
        return 0
    fi
    
    # Download file using wget or curl
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$output_path" "$file_url"
    elif command -v curl &> /dev/null; then
        curl -L -# -o "$output_path" "$file_url"
    else
        echo "   ❌ Error: Neither wget nor curl is available. Please install one of them."
        return 1
    fi
    
    # Check if download was successful
    if [[ $? -eq 0 ]] && [[ -f "$output_path" ]]; then
        echo "   ✅ Successfully downloaded: $file_name"
        return 0
    else
        echo "   ❌ Failed to download: $file_name"
        return 1
    fi
}

#######################################################################
# Download Inference Data Files
#######################################################################

echo "-----------------------------------------------------------"
echo "📦 Downloading Inference Data Files"
echo "-----------------------------------------------------------"

for file_url in "${INFERENCE_FILES[@]}"; do
    file_name=$(basename "$file_url")
    download_file "$file_name" "$file_url"
    echo ""
done

#######################################################################
# Download Training Data Files (if requested)
#######################################################################

if [[ "$inference_only" != "true" ]]; then
    echo ""
    echo "-----------------------------------------------------------"
    echo "📦 Downloading Training Data Files"
    echo "-----------------------------------------------------------"

    for file_url in "${TRAINING_FILES[@]}"; do
        file_name=$(basename "$file_url")
        download_file "$file_name" "$file_url"
        echo ""
    done
else
    echo ""
    echo "ℹ️  Skipping training data download (inference_only=true)"
fi

#######################################################################
# Data Verification
#######################################################################

echo ""
echo "-----------------------------------------------------------"
echo "🔍 Verifying Downloaded Files"
echo "-----------------------------------------------------------"

all_files_exist=true

for file_url in "${INFERENCE_FILES[@]}"; do
    file_name=$(basename "$file_url")
    file_path="$data_root_dir/$file_name"
    
    if [[ -f "$file_path" ]]; then
        file_size=$(du -h "$file_path" | cut -f1)
        echo "✅ $file_name: $file_size"
    else
        echo "❌ $file_name: MISSING"
        all_files_exist=false
    fi
done

if [[ "$inference_only" != "true" ]]; then
    for file_url in "${TRAINING_FILES[@]}"; do
        file_name=$(basename "$file_url")
        file_path="$data_root_dir/$file_name"
        
        if [[ -f "$file_path" ]]; then
            file_size=$(du -h "$file_path" | cut -f1)
            echo "✅ $file_name: $file_size"
        else
            echo "❌ $file_name: MISSING"
            all_files_exist=false
        fi
    done
fi

echo ""
echo "-----------------------------------------------------------"

if [[ "$all_files_exist" == "true" ]]; then
    echo "🎉 SUCCESS: All required data files have been downloaded!"
    echo "   Data location: $data_root_dir"
else
    echo "⚠️ WARNING: Some files are missing. Please check the download logs above."
    exit 1
fi

echo "-----------------------------------------------------------"
