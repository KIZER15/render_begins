# Use a stable and compatible base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Check if the model is copied
RUN ls -l /app

# Create upload folder
RUN mkdir -p uploads

# Expose the default Flask port
EXPOSE 5000


# Replace the last line in Dockerfile
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
