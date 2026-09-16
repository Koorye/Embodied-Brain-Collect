.PHONY: format
format:
	uvx pre-commit run -a

.PHONY: type
type:
	uvx ty check

.PHONY: check
check: format type

.PHONY: test
test:
	uv run --extra dev pytest

.PHONY: build
build:
	uv build
