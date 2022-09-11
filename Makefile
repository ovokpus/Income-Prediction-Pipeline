test:
	pytest tests/


quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	docker-compose build

	docker-compose up

publish: build
