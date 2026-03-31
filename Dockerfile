# ─────────────────────────────────────────────────────────────
#  TechBot – AI Chatbot Dockerfile
#  Tests the trained ML model in a Linux container.
#
#  NOTE: WinForms GUI cannot run in Linux Docker.
#  This container runs the Python ML model in headless/CLI mode,
#  which proves the AI model works inside Docker.
#
#  Build:  docker build -t techbot .
#  Test:   docker run --rm techbot python3 Python/predict.py "what is AI"
# ─────────────────────────────────────────────────────────────

# ── Stage 1: Build the .NET app ───────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build

WORKDIR /src

# Restore dependencies first (layer caching)
COPY AIChatbot/AIChatbot.csproj ./AIChatbot/
RUN dotnet restore ./AIChatbot/AIChatbot.csproj

# Copy and publish the app
COPY AIChatbot/ ./AIChatbot/
RUN dotnet publish ./AIChatbot/AIChatbot.csproj \
    -c Release \
    -o /app/publish \
    --no-restore

# ── Stage 2: Runtime with Python for ML inference ─────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install Python ML packages
RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    numpy==1.26.2

# Copy trained model files
COPY AIChatbot/Model/ ./Model/

# Copy Python inference script
COPY AIChatbot/Python/ ./Python/

# Copy training script (for reference/documentation)
COPY TrainingScript/ ./TrainingScript/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command: run a demo prediction to prove the model works
CMD ["python3", "Python/predict.py", "what is machine learning"]
