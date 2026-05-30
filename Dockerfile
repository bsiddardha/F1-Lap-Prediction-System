# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages only
COPY --from=builder /install /usr/local

COPY app.py .
COPY templates ./templates
COPY f1_random_forest.pkl .
COPY f1_feature_columns.pkl .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]