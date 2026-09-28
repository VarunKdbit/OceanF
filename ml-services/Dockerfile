FROM python:3.11-slim

WORKDIR /workspace/ml-services

COPY ml-services/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml-services/requirements-ingestion.txt /tmp/requirements-ingestion.txt
RUN python -m venv /opt/oceanembed-ingestion \
    && /opt/oceanembed-ingestion/bin/pip install --no-cache-dir -r /tmp/requirements-ingestion.txt

COPY ml-services/app ./app
COPY scripts /workspace/scripts

ENV OCEANEMBED_INGESTION_PYTHON=/opt/oceanembed-ingestion/bin/python

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
