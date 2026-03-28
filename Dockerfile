FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by pytesseract and xgboost
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the standard port for Hugging Face Spaces
EXPOSE 7860

# Run the application using Gunicorn (we run on port 7860)
CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:7860"]
