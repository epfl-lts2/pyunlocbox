.PHONY: clean doc lint test release dist

help:
	@echo "clean    remove non-source files"
	@echo "doc      generate Sphinx HTML documentation, including API doc"
	@echo "lint     check style with flake8"
	@echo "test     run tests and check code coverage"
	@echo "release  package and upload a release"
	@echo "dist     package"

clean:
	# Python files.
	find . -name '__pycache__' -exec rm -rf {} +
	# Documentation.
	rm -rf doc/_build
	# Coverage.
	rm -rf .coverage
	rm -rf htmlcov
	# Package build.
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info

doc:
	$(MAKE) -C doc html

lint:
	flake8

test:
	coverage run --branch --source pyunlocbox setup.py test
	coverage report
	coverage html

release: clean
	python setup.py register
	python setup.py sdist upload
#	python setup.py bdist_wheel upload

dist: clean
	python setup.py sdist
#	python setup.py bdist_wheel
	ls -l dist
