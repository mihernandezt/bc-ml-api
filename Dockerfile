# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ model/
COPY app/ app/

EXPOSE 5000

CMD ["python", "-m", "app.main"]