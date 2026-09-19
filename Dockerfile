FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so they layer-cache independently of source changes.
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir .

COPY sample_data/ sample_data/
COPY eval/ eval/
COPY scripts/ scripts/

EXPOSE 8000
CMD ["uvicorn", "rag_qa.api:app", "--host", "0.0.0.0", "--port", "8000"]
