# Start from a base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the necessary files into the container
COPY . ./

# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Ensure Cloud Run uses the correct port
ENV PORT=8080
EXPOSE 8080

# Run Uvicorn on the correct port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]