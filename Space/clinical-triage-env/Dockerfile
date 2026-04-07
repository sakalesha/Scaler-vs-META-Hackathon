FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# HF Spaces runs as port 7860
EXPOSE 7860

# HF_TOKEN will be injected as a Space secret — do NOT hardcode it
ENV HF_TOKEN=""
ENV PYTHONUNBUFFERED=1

# Launch the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
