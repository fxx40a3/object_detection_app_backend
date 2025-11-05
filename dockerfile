FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for psycopg2 and torch
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code
COPY backend /app

# Copy requirements file
COPY backend/requirements.txt /app/

# Install dependencies from PyPI
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]