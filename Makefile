test:
	pytest tests/


quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	# LOCAL_TAG=`date + "%Y-%m-%d-%H-%M"`
	# LOCAL_IMAGE_NAME=stre

publish: build
