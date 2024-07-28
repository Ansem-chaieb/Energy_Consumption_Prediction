FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY .env /app/

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]