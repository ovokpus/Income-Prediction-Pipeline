FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PREDICTION_SERVICE_URL=http://prediction_service:9696

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY features.py app.py ./
COPY .streamlit ./.streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
