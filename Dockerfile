ARG PYTHON_VERSION=3.12.1
FROM python:${PYTHON_VERSION}-slim-bullseye as base 

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Sets the working directory inside the container to /app.
WORKDIR /app

# Install system dependencies for PostgreSQL and Psycopg3
RUN apt-get update && apt-get install -y \
    libpq-dev gcc python3-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 5050

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5050",]

