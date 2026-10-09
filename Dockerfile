# Dockerfile
# Packages the prediction API and trained model into a container (Lesson 39 Demo 2)
# Build from the project folder with: docker build -t titanic-api:v1 .

# Use the official slim Python image, matching the Python version used on the Mac
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the API's pinned package list and install it
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the API code and the trained model into the container
COPY app.py .
COPY model.pkl .

# Tell Docker the API listens on port 8000
EXPOSE 8000

# Start the API when the container runs; 0.0.0.0 lets the Mac reach it from outside the container
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]