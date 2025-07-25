FROM python:3.12-slim

# Install Tesseract OCR and wget
RUN apt-get update && apt-get install -y tesseract-ocr=5.3.0-2 wget && rm -rf /var/lib/apt/lists/*

# Set the Tesseract data path (important for finding training data)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Download additional language data (cached independently)
RUN wget https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata -P ${TESSDATA_PREFIX}
RUN wget https://github.com/tesseract-ocr/tessdata_best/raw/main/hin.traineddata -P ${TESSDATA_PREFIX}

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY ./src/requirements.txt .

# Install Python dependencies (this layer will be cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY ./src .

# Declare volume for output files
VOLUME ["/app/data"]

# Use ENTRYPOINT to allow arguments to be passed to the Python script
ENTRYPOINT ["python", "main.py"]