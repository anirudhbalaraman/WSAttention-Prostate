.PHONY: format lint typecheck test check

format:
	ruff format .

lint:
	ruff check . --fix

typecheck:
	mypy .

test:
	pytest

clean:
	@echo "Cleaning project..."
	# Delete compiled bytecode
	@python3 -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	# Delete directory-based caches
	@python3 -Bc "import shutil, pathlib; \
		[shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; \
		[shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ipynb_checkpoints')]; \
		[shutil.rmtree(p) for p in pathlib.Path('.').rglob('.monai-cache')]; \
		[shutil.rmtree(p) for p in pathlib.Path('.').rglob('.mypy_cache')]; \
		[shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ruff_cache')]; \
		[shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"

# Updated 'check' to clean before running (optional)
# This ensures you are testing from a "blank slate"
check: format lint typecheck test clean