# Tesseract OCR Docker Container

This Docker container provides an easy way to run Tesseract OCR on images using Python.

## Building the Container

```bash
docker build -t tesseract-ocr .
```

## Running the Container with Arguments

The container accepts command line arguments for the Python script. The basic syntax is:

```bash
docker run --rm tesseract-ocr <image_path> [OPTIONS]
```

### Arguments

- `image_path`: Path to the image file to process (required)
- `--lang` or `-l`: Language for OCR (default: eng)
- `--output-type` or `-o`: Output type - 'text' (default) or 'data' (with bounding boxes)
- `--save-file` or `-s`: Save OCR result to a text file in data directory with same base name as image

### Examples

1. **Basic OCR with default settings (English, text output):**

   ```bash
   docker run --rm tesseract-ocr data/ocr-example.png
   ```

2. **Specify language:**

   ```bash
   docker run --rm tesseract-ocr data/ocr-example.png --lang fra
   ```

3. **Get detailed data with bounding boxes:**

   ```bash
   docker run --rm tesseract-ocr data/ocr-example.png --output-type data
   ```

4. **Using short form arguments:**

   ```bash
   docker run --rm tesseract-ocr data/ocr-example.png -l spa -o data
   ```

5. **Save OCR result to file:**

   ```bash
   docker run --rm tesseract-ocr data/ocr-example.png --save-file
   ```

6. **Combine multiple options:**

   ```bash
   docker run --rm tesseract-ocr data/ocr-example.png --lang fra --save-file
   ```

7. **Show help:**
   ```bash
   docker run --rm tesseract-ocr --help
   ```

### Processing External Images

To process images from your local machine, mount a volume:

```bash
docker run --rm -v /path/to/your/images:/images tesseract-ocr /images/your-image.jpg
```

Example:

```bash
docker run --rm -v $(pwd)/my-images:/images tesseract-ocr /images/document.png --lang eng
```

### Available Languages

Common language codes:

- `eng` - English (default)
- `fra` - French
- `deu` - German
- `spa` - Spanish
- `ita` - Italian
- `por` - Portuguese
- `rus` - Russian
- `chi_sim` - Chinese Simplified
- `chi_tra` - Chinese Traditional
- `jpn` - Japanese
- `kor` - Korean

## Quick Start

Run the provided examples script:

```bash
./run_docker_examples.sh
```

This will build the container and run several example commands to demonstrate the functionality.
