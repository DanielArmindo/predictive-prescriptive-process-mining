.PHONY: test

ifeq ($(OS),Windows_NT)
OPEN_CMD = start ""
else ifeq ($(shell uname),Darwin)
OPEN_CMD = open
else
OPEN_CMD = xdg-open
endif


dev:
	uv run fastapi dev main.py

run:
	uv run fastapi run main.py

test:
	uv run python -m test.main

endpoints:
	$(OPEN_CMD) http://127.0.0.1:8000/docs
