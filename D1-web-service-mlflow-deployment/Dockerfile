FROM python:3.8.10-slim

RUN pip install -U pip 

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "../models/", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn" "--bind=localhost:9696" "predict:app"]



