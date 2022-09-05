FROM python:3.8-slim

RUN pip install -U pip 

RUN pip install pipenv

ENV MODEL_RUN_ID="434e8043496c4d61ae917c1f0ef843ba"

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]

COPY ["models/", "models/"]

# COPY ["mlruns/", "mlruns/"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--timeout=1000", "--bind=0.0.0.0:9696", "predict:app"]



