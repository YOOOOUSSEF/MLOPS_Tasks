FROM python:3.12-slim

WORKDIR /app

COPY requirements/runtime.txt requirements/runtime.txt
RUN pip install --no-cache-dir -r requirements/runtime.txt

COPY app/ app/
COPY src/ src/
COPY config/ config/
COPY data/raw/ML_TABLE_LABELED.csv data/raw/ML_TABLE_LABELED.csv
COPY models/ models/
COPY feature_list.txt .

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]