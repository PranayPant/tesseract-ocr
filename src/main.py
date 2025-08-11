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

lang = "eng"  # Default language for OCR
config = "--tessdata-dir /usr/share/tesseract-ocr/5/tessdata --oem 3 --psm 3"

# Global directories object - initialized in main()
DIRS = None


def setup_directories(base_name):
    """
    Create all necessary directories for processing and return path objects
    """
    base_dir = Path("data")

    # Define all directory paths
    directories = {
        "base": base_dir,
        "preprocessed": base_dir / "preprocessed" / base_name,
        "roi": base_dir / "roi" / base_name,
        "results": base_dir / "results",
    }

    # Create all directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def get_base_info(image_path):
    image_path_obj = Path(image_path)
    base_name = image_path_obj.stem
    ext = image_path_obj.suffix

    return base_name, ext


def find_text_regions(binary, gray, image_path):

    img = cv2.imread(image_path)
    original_img_for_drawing = img.copy()
    base_name, ext = get_base_info(image_path)

    # List to store all OCR results
    ocr_results = []

    # 2. Morphological Operations to connect characters into words/lines
    # Define a horizontal kernel to connect characters within a line.
    # Adjust kernel size as needed: (width, height)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # 3. Find contours on the dilated image
    raw_contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # --- NEW: Sort contours by their position (top-to-bottom, then left-to-right) ---
    # Create a list of (contour, bounding_box_y, bounding_box_x) tuples
    sorted_contours_with_pos = []
    for contour in raw_contours:
        x, y, w, h = cv2.boundingRect(contour)
        sorted_contours_with_pos.append(
            (contour, y, x)
        )  # Store contour, then y, then x

    # Sort this list based on y-coordinate, then x-coordinate
    sorted_contours_with_pos.sort(
        key=lambda item: (item[1], item[2])
    )  # Sort by y, then x

    # Extract just the sorted contours
    contours = [item[0] for item in sorted_contours_with_pos]
    # --- END NEW SORTING ---

    confirmed_text_regions = []

    # Define criteria for filtering text regions (these are empirical and may need tuning)
    min_area = 100
    max_area = img.shape[0] * img.shape[1] / 2
    min_text_height = 10
    max_text_height = 200

    min_text_aspect_ratio = 1.0
    max_text_aspect_ratio = 20.0

    max_line_thickness = 10
    min_line_length = 50

    min_text_solidity = 0.5
    min_text_extent = 0.3

    # Tesseract configuration for OCR on regions
    # PSM 3: Fully automatic page segmentation, but no OSD. Good for general text blocks.
    ocr_config = (
        f"--psm 3 --oem 3 --tessdata-dir /usr/share/tesseract-ocr/5/tessdata"
    )

    # 4. Filter and Refine Text Regions with OCR Validation
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Skip very small contours (likely noise) early
        if area < min_area:
            continue

        aspect_ratio = w / h if h > 0 else 0

        # Calculate solidity and extent
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0

        # --- Filtering Logic ---
        is_potential_text_by_shape = False

        if (
            area > min_area
            and area
            < max_area  # Corrected: max_area was missing from the condition
            and min_text_height < h < max_text_height
            and min_text_aspect_ratio < aspect_ratio < max_text_aspect_ratio
            and w > 20
        ):  # Ensure sufficient width for a text line
            is_potential_text_by_shape = True

        # 2. Check for line characteristics (very thin and elongated)
        is_horizontal_line = h < max_line_thickness and w > min_line_length
        is_vertical_line = w < max_line_thickness and h > min_line_length

        if is_horizontal_line or is_vertical_line:
            # If it looks like a line, it's NOT text, regardless of other properties
            cv2.rectangle(
                original_img_for_drawing, (x, y), (x + w, y + h), (0, 0, 255), 1
            )  # Red for lines
            cv2.putText(
                original_img_for_drawing,
                "Line",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            continue  # Skip this contour, it's a line

        # 3. Use solidity and extent to further confirm text-like blobs
        if (
            is_potential_text_by_shape
            and solidity > min_text_solidity
            and extent > min_text_extent
        ):
            # This contour is a potential text region based on its shape.
            # Now, validate it with OCR.

            # Extract ROI from the original grayscale image for OCR
            # Using grayscale ROI for Tesseract is often better, as it handles its own binarization
            roi_gray = gray[y : y + h, x : x + w]
            roi_path = DIRS["roi"] / f"{base_name}_roi_{i}_{x}x{y}{ext}"
            cv2.imwrite(str(roi_path), roi_gray)

            # Perform OCR
            ocr_text = pytesseract.image_to_string(
                roi_gray, lang="hin", config=ocr_config
            )

            # Clean up OCR text (remove common noise characters or very short strings)
            # You might need to refine this cleaning based on your specific text content
            # cleaned_ocr_text = ''.join(
            #     char for char in ocr_text if char.isalnum() or char.isspace()).strip()
            cleaned_ocr_text = (
                ocr_text.replace("\n", " ").replace("\r", "").strip()
            )

            # Store OCR result for this ROI (regardless of validation outcome)
            ocr_results.append(
                {
                    "roi_index": i,
                    "text": (
                        cleaned_ocr_text
                        if cleaned_ocr_text
                        else "[No text detected]"
                    ),
                    "coordinates": (x, y, w, h),
                }
            )

            # OCR Validation: Check if the detected text is meaningful
            # Criteria: non-empty, and has at least 2 alphanumeric characters (adjust as needed)
            # or check for specific character sets
            if cleaned_ocr_text and len(cleaned_ocr_text) > 1:
                confirmed_text_regions.append((x, y, w, h, cleaned_ocr_text))
                # Draw the bounding box and detected text
                # Green for confirmed text
                cv2.rectangle(
                    original_img_for_drawing,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    5,
                )
                # cv2.putText(
                #     original_img_for_drawing,
                #     cleaned_ocr_text,
                #     (x, y - 5),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,
                #     (0, 165, 255),
                #     2,
                # )
            else:
                # It looked like text by shape, but OCR failed to find meaningful text.
                # Re-classify as non-text.
                # Blue for OCR-rejected
                cv2.rectangle(
                    original_img_for_drawing,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    5,
                )
                cv2.putText(
                    original_img_for_drawing,
                    "Non-Text (OCR Fail)",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
        else:
            # This contour didn't even pass the initial shape-based text criteria
            # Red for other non-text
            cv2.rectangle(
                original_img_for_drawing, (x, y), (x + w, y + h), (0, 0, 255), 5
            )
            cv2.putText(
                original_img_for_drawing,
                "Other Non-Text",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    # Save the image with detected text regions for debugging
    text_regions_path = DIRS["preprocessed"] / f"{base_name}_text_regions{ext}"
    cv2.imwrite(str(text_regions_path), original_img_for_drawing)

    # Save all OCR results to a single text file
    save_roi_ocr_results(base_name, ocr_results)

    return confirmed_text_regions, original_img_for_drawing


def save_roi_ocr_results(base_name, ocr_results):
    """Save all ROI OCR results to a single text file"""
    output_filename = f"{base_name}_roi_ocr_results.txt"
    output_path = DIRS["results"] / output_filename

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"OCR Results for {base_name}\n")
            f.write("=" * 50 + "\n\n")

            for result in ocr_results:
                roi_index = result["roi_index"]
                text = result["text"]
                coords = result["coordinates"]

                f.write(f"[ {roi_index} ]\n")
                f.write(
                    f"Coordinates: x={coords[0]}, y={coords[1]}, w={coords[2]}, h={coords[3]}\n"
                )
                f.write(f"{text}\n\n")

        print(f"ROI OCR results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving ROI OCR results: {e}")


def preprocess_image(image_path):
    """
    Preprocess image using OpenCV to improve OCR accuracy
    """
    # Read image using OpenCV
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    base_name, ext = get_base_info(image_path)

    denoisedColor = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv2.imwrite(
        str(DIRS["preprocessed"] / f"{base_name}_01_denoised{ext}"),
        denoisedColor,
    )

    # Convert to grayscale
    gray = cv2.cvtColor(denoisedColor, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(DIRS["preprocessed"] / f"{base_name}_02_gray{ext}"), gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(
        str(DIRS["preprocessed"] / f"{base_name}_03_blurred{ext}"), blurred
    )

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    cv2.imwrite(
        str(DIRS["preprocessed"] / f"{base_name}_04_sharpened{ext}"),
        sharpened,
    )

    # Can also apply adaptive thresholding to get better contrast
    # This works better than simple thresholding for varying lighting conditions
    # Use regular thresholding instead for now
    thresh = cv2.threshold(
        sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    cv2.imwrite(
        str(DIRS["preprocessed"] / f"{base_name}_05_thresh{ext}"), thresh
    )

    # Optional: Apply morphological operations to clean up the image
    # kernel = np.ones((2, 2), np.uint8)
    # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    find_text_regions(thresh, gray, image_path)

    # Convert back to PIL Image format for pytesseract
    # Make sure DPI is at least 300 for better OCR results
    pil_image = Image.fromarray(thresh)
    # Save with 300 DPI
    pil_image.save(
        str(DIRS["preprocessed"] / f"{base_name}_05_thresh{ext}"),
        dpi=(300, 300),
    )


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Extract text from images using Tesseract OCR"
    )
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument(
        "--lang", "-l", default="eng", help="Language for OCR (default: eng)"
    )
    parser.add_argument(
        "--psm",
        "--page-segmentation-mode",
        type=int,
        default=3,
        choices=range(0, 14),
        help="Page segmentation mode for Tesseract (0-13, default: 3 - Fully automatic page segmentation)",
    )
    parser.add_argument(
        "--save-file",
        "-s",
        action="store_true",
        help="Save OCR result to a text file in data directory with same base name as image",
    )

    args = parser.parse_args()
    lang = args.lang
    config = f"--tessdata-dir /usr/share/tesseract-ocr/5/tessdata --oem 3 --psm {args.psm}"

    # Preprocess and load the image
    try:
        # Setup directories once for the entire process
        base_name, ext = get_base_info(args.image_path)
        global DIRS
        DIRS = setup_directories(base_name)

        print(f"Preprocessing image: {args.image_path}")
        preprocess_image(args.image_path)
        print("Image preprocessing completed")

        # Always load the image with PIL for Tesseract
        final_image_path = DIRS["preprocessed"] / f"{base_name}_05_thresh{ext}"
        img = Image.open(str(final_image_path))
        print(f"Processing image: {final_image_path}")
    except FileNotFoundError:
        print(
            f"Error: '{args.image_path}' not found. Please provide a valid image file."
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error opening/processing image: {e}")
        sys.exit(1)

    text = pytesseract.image_to_string(img, lang=args.lang, config=config)

    # Save to file if requested
    if args.save_file:
        save_text_to_file(args.image_path, text)
    else:
        print("Extracted Text:")
        print(text)


def save_text_to_file(image_path, text_content):
    """Save OCR text content to a file in the results directory with same base name as image"""
    # Get the base name of the image file without extension
    image_path_obj = Path(image_path)
    base_name = image_path_obj.stem

    # Create output filename with .txt extension
    output_filename = f"{base_name}_ocr_result.txt"

    # Use global directories
    output_path = DIRS["results"] / output_filename

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"OCR result saved to: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
