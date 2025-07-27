from PIL import Image
import pytesseract
import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
import shutil

# If Tesseract is not in your system's PATH, you need to specify its path.
# For example, on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux/macOS, it's usually found automatically if installed correctly.


def load_image(image_path):
    """
    Load image using OpenCV
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img


def denoise_image(img):
    """
    Remove noise from colored image using Non-local Means Denoising
    """
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def convert_to_grayscale(img):
    """
    Convert image to grayscale
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(img, kernel_size=(5, 5)):
    """
    Apply Gaussian blur to reduce noise
    """
    return cv2.GaussianBlur(img, kernel_size, 0)


def sharpen_image(img):
    """
    Apply sharpening filter to enhance edges
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def apply_threshold(img):
    """
    Apply OTSU thresholding to get binary image
    """
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def apply_morphological_operations(img, kernel_size=(2, 2)):
    """
    Apply morphological operations to clean up the image (optional)
    """
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def save_preprocessed_image(img, image_path, dpi=(300, 300)):
    """
    Convert processed image back to PIL format and save with specified DPI
    """
    pil_image = Image.fromarray(img)
    pil_image.save(image_path, dpi=dpi)


def save_step_image(img, original_path, step_name):
    """
    Save intermediate preprocessing step with unique filename
    """
    path_obj = Path(original_path)
    base_name = path_obj.stem
    extension = path_obj.suffix
    parent_dir = path_obj.parent

    step_filename = f"{base_name}_{step_name}{extension}"
    step_path = parent_dir / step_filename

    # Convert to PIL format and save
    pil_image = Image.fromarray(img)
    pil_image.save(step_path, dpi=(300, 300))
    print(f"Saved {step_name} step: {step_path}")


def preprocess_image(image_path):
    """
    Preprocess image using OpenCV to improve OCR accuracy
    Applies the full preprocessing pipeline and saves each step
    """
    # Create a copy of the original image to work on
    path_obj = Path(image_path)
    base_name = path_obj.stem
    extension = path_obj.suffix
    parent_dir = path_obj.parent

    # Create working copy filename
    working_copy_path = parent_dir / f"{base_name}_working_copy{extension}"

    # Copy the original image to working copy
    shutil.copy2(image_path, working_copy_path)
    print(f"Created working copy: {working_copy_path}")

    # Load the working copy image (original remains untouched)
    img = load_image(str(working_copy_path))
    save_step_image(img, image_path, "01_original")

    # Apply denoising
    img = denoise_image(img)
    save_step_image(img, image_path, "02_denoised")

    # Convert to grayscale
    img = convert_to_grayscale(img)
    save_step_image(img, image_path, "03_grayscale")

    # Apply Gaussian blur to reduce noise
    img = apply_gaussian_blur(img)
    save_step_image(img, image_path, "04_blurred")

    # Sharpen the image
    img = sharpen_image(img)
    save_step_image(img, image_path, "05_sharpened")

    # Apply thresholding
    img = apply_threshold(img)
    save_step_image(img, image_path, "06_thresholded")

    # Optional: Apply morphological operations to clean up the image
    # img = apply_morphological_operations(img)
    # save_step_image(img, image_path, "07_morphological")

    # Save the final preprocessed image to the working copy
    save_preprocessed_image(img, working_copy_path)
    print(f"Final preprocessed image saved to: {working_copy_path}")

    return working_copy_path  # Return the path to the working copy with final processing


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
        final_processed_path = preprocess_image(args.image_path)
        print("Image preprocessing completed")

        # Load the final processed image with PIL for Tesseract
        img = Image.open(final_processed_path)
        print(f"Processing final image: {final_processed_path}")
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
