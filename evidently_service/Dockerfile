# syntax=docker/dockerfile:1

FROM python:3.8-slim-bullseye

WORKDIR /app

# evidently 0.1.51 is the last API this service imports
# (evidently.model_monitoring). Its own pin of PyYAML~=5.1 does not ship a
# wheel we can build here, so the library is installed with --no-deps and
# the requirements file supplies the dependencies that do install.
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir --no-deps evidently==0.1.51.dev0

COPY app.py config.yaml ./
COPY datasets ./datasets

EXPOSE 8085

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=8085"]
