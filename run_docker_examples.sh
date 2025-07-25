#!/bin/bash

# Build the Docker i# Example 3: Run OCR and save result to file
echo "Example 3: Running OCR and saving result to file"
docker run --rm tesseract-ocr data/ocr-example.png --save-file

echo ""
echo "============================="
echo ""

# Example 4: Run OCR with language and save to file
echo "Example 4: Running OCR with Spanish language and saving to file"
docker run --rm tesseract-ocr data/ocr-example-2.jpg --lang spa --save-fileg D# Example 7: Run OCR with different PSM mode
echo "Example 7: Running OCR with PSM 7 (single line text)"
docker run --rm tesseract-ocr data/ocr-example.png --psm 7

echo ""
echo "============================="
echo ""

# Example 8: Run OCR with default preprocessing and custom PSM
echo "Example 8: Running OCR with default PSM 3 and custom language"
docker run --rm tesseract-ocr data/ocr-example.png --lang eng..."
docker build --rm -t tesseract-ocr .

echo "Docker image built successfully!"
echo ""

# Example 1: Run OCR on the first image with default settings
echo "Example 1: Running OCR on ocr-example.png with default settings (English, text output, preprocessing always enabled)"
docker run --rm tesseract-ocr data/ocr-example.png

echo ""
echo "============================="
echo ""

# Example 2: Run OCR on the second image with different language
echo "Example 2: Running OCR on ocr-example-2.jpg with English language specified"
docker run --rm tesseract-ocr data/ocr-example-2.jpg --lang eng

echo ""
echo "============================="
echo ""

# Example 3: Run OCR with data output (bounding boxes)
echo "Example 3: Running OCR with data output (bounding boxes)"
docker run --rm tesseract-ocr data/ocr-example.png --output-type data

echo ""
echo "============================="
echo ""

# Example 4: Run OCR and save result to file
echo "Example 4: Running OCR and saving result to file"
docker run --rm tesseract-ocr data/ocr-example.png --save-file

echo ""
echo "============================="
echo ""

# Example 5: Run OCR with Spanish language and saving to file
echo "Example 5: Running OCR with Spanish language and saving to file"
docker run --rm tesseract-ocr data/ocr-example-2.jpg --lang spa --save-file

echo ""
echo "============================="
echo ""

# Example 6: Run OCR with custom PSM (Page Segmentation Mode)
echo "Example 6: Running OCR with PSM 8 (single word)"
docker run --rm tesseract-ocr data/ocr-example.png --psm 8

echo ""
echo "============================="
echo ""

# Example 7: Run OCR with PSM 7 for single line text
echo "Example 7: Running OCR with PSM 7 (single line text)"
docker run --rm tesseract-ocr data/ocr-example.png --psm 7

echo ""
echo "============================="
echo ""

# Example 8: Show help
echo "Example 8: Showing help message"
docker run --rm tesseract-ocr --help
