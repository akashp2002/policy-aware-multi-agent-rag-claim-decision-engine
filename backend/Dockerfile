FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data
COPY custom_cases ./custom_cases

# Build the policy chunk index (fastembed downloads ONNX models on first run).
RUN python -c "from src.retrieval.policy_ingestion import ingest_policy; ingest_policy(force=False)"

RUN useradd --create-home --uid 10001 appuser \
	&& chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]