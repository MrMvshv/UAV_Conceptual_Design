FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set up SUAVE
RUN pip install numpy==1.21.6 scipy==1.7.3 matplotlib==3.5.3
RUN git clone https://github.com/suavecode/SUAVE.git && \
    cd SUAVE && \
    pip install -e .

WORKDIR /app