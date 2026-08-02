# Use the Python 3.10 slim base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy all files from the current directory to the working directory in the container
COPY ./ .

# Define the virtual environment path
ENV VIRTUAL_ENV=/opt/venv

# Create a virtual environment
RUN python3 -m venv $VIRTUAL_ENV

# Add the virtual environment's binary path to the system PATH
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip inside the virtual environment
RUN pip install --upgrade pip

# Install the package from the current directory
RUN pip install . --no-cache-dir

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Probe /health; python stdlib avoids adding curl to the slim image just for this.
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"


# Start the FastAPI application using uvicorn
CMD ["uvicorn", "src.app.main:app", "--workers", "4", "--host", "0.0.0.0", "--port", "8000"]
