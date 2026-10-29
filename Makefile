test:
	pytest tests/


quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	docker compose build

up:
	docker compose up --build

publish: build

setup:
	pip install -r requirements.txt
	pre-commit install
