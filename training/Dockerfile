# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Cython separately to resolve the issue
RUN pip install --no-cache-dir Cython

# Install any other needed packages specified in requirements.txt
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Train the model
RUN python3 training/train.py
