VENV=venv
PYTHON=$(VENV)/bin/python3
FILES=$(shell git ls-files '*.py')

build: requirements.txt
	@if [ ! -d $(VENV) ]; then virtualenv -p python3 $(VENV); fi
	@$(PYTHON) -m pip install -r requirements.txt;
	@$(PYTHON) sys_check.py

format:
	@$(PYTHON) -m black .

run:
	@$(PYTHON) src/detect.py

lint:
	@$(PYTHON) -m black --diff --check $(FILES)
	@$(PYTHON) -m pylint --disable=all --enable=unused-import $(FILES)

clean:
	rm -rf .mypy_cache
	rm -rf runs
