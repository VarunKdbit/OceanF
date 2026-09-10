FROM python:3.11-slim

WORKDIR /workspace/ml-services

COPY ml-services/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml-services/app ./app
COPY scripts /workspace/scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
