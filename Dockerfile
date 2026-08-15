FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir \
    "pydantic>=2" \
    "langgraph>=0.6.7" \
    "tenacity>=9.1.2" \
    "arxiv>=2.2.0" \
    "openai>=1.30.0" \
    "python-dotenv>=1.1.1" \
    "pyyaml>=6.0.2" \
    "pymupdf==1.28.0" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "httpx"

COPY . .

EXPOSE 8000

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}