.PHONY: help clean lint test doc dist release

help:
	@echo "clean    remove non-source files"
	@echo "lint     check style"
	@echo "test     run tests and check coverage"
	@echo "doc      generate HTML documentation and check links"
	@echo "dist     package (source & wheel)"
	@echo "release  package and upload to PyPI"

clean:
	git clean -Xdf

lint:
	flake8 --doctests --exclude=doc

# Matplotlib doesn't print to screen. Also faster.
export MPLBACKEND = agg

test:
	coverage run --branch --source pyunlocbox setup.py test
	coverage report
	coverage html

doc:
	sphinx-build -b html -d doc/_build/doctrees doc doc/_build/html
	sphinx-build -b linkcheck -d doc/_build/doctrees doc doc/_build/linkcheck

dist: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	ls -l dist

release: dist
	twine upload dist/*
