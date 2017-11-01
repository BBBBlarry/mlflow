PROJECT_NAME = mlflow

init:
	pip install -r requirements.txt
build: 
	python setup.py build
install: 
	python setup.py install
example: 
	python ./examples/example.py
clean-build:
	rm -f -r build/
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
test:
	pytest -q tests
clean-test-cache:
	find . -name '.cache' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +

isort:
	sh -c "isort --recursive . "

help:
	@echo "    init"
	@echo "        Install requirements."
	@echo "    build"
	@echo "        Build ${PROJECT_NAME}."
	@echo "    install"
	@echo "        Install ${PROJECT_NAME} on your local machine."
	@echo "    example"
	@echo "        Run example."
	@echo "    test"
	@echo "        Run the test suite."
	@echo "    clean-build"
	@echo "        Remove build artifacts."
	@echo "    clean-pyc"
	@echo "        Remove python artifacts."
	@echo "    clean-test-cache"
	@echo "        Remove test cache."
	@echo "    isort"
	@echo "        Sort import statements."

.PHONY: test clean-build clean-pyc clean-test-cache