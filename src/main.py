from PIL import Image
import pytesseract
import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

# If Tesseract is not in your system's PATH, you need to specify its path.
# For example, on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux/macOS, it's usually found automatically if installed correctly.


def preprocess_image(image_path):
    """
    Preprocess image using OpenCV to improve OCR accuracy
    """
    # Read image using OpenCV
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    denoisedColor = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Convert to grayscale
    gray = cv2.cvtColor(denoisedColor, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)

    # Can also apply adaptive thresholding to get better contrast
    # This works better than simple thresholding for varying lighting conditions
    # Use regular thresholding instead for now
    thresh = cv2.threshold(
        sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Optional: Apply morphological operations to clean up the image
    # kernel = np.ones((2, 2), np.uint8)
    # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Convert back to PIL Image format for pytesseract
    # Make sure DPI is at least 300 for better OCR results
    pil_image = Image.fromarray(thresh)
    pil_image.save(image_path, dpi=(300, 300))  # Save with 300 DPI


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Extract text from images using Tesseract OCR')
    parser.add_argument('image_path', help='Path to the image file to process')
    parser.add_argument('--lang', '-l', default='eng',
                        help='Language for OCR (default: eng)')
    parser.add_argument('--psm', '--page-segmentation-mode', type=int, default=3,
                        choices=range(0, 14),
                        help='Page segmentation mode for Tesseract (0-13, default: 3 - Fully automatic page segmentation)')
    parser.add_argument('--save-file', '-s', action='store_true',
                        help='Save OCR result to a text file in data directory with same base name as image')

    args = parser.parse_args()

    # Preprocess and load the image
    try:
        print(f"Preprocessing image: {args.image_path}")
        preprocess_image(args.image_path)
        print("Image preprocessing completed")

        # Always load the image with PIL for Tesseract
        img = Image.open(args.image_path)
        print(f"Processing image: {args.image_path}")
    except FileNotFoundError:
        print(
            f"Error: '{args.image_path}' not found. Please provide a valid image file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error opening/processing image: {e}")
        sys.exit(1)

    config = f'--tessdata-dir /usr/share/tesseract-ocr/5/tessdata --oem 3 --psm {args.psm}'

    text = pytesseract.image_to_string(
        img, lang=args.lang, config=config)
    print("Extracted Text:")
    print(text)

    # Save to file if requested
    if args.save_file:
        save_text_to_file(args.image_path, text)


def save_text_to_file(image_path, text_content):
    """Save OCR text content to a file in the data directory with same base name as image"""
    # Get the base name of the image file without extension
    image_path_obj = Path(image_path)
    base_name = image_path_obj.stem

    # Create output filename with .txt extension
    output_filename = f"{base_name}_ocr_result.txt"

    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Full output path
    output_path = data_dir / output_filename

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        print(f"OCR result saved to: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
