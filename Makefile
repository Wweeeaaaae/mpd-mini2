package: 
	python setup.py sdist

install: 
	pip install $(shell ls dist/*.tar.gz)

all: clean package install

upload:
	python -m twine upload dist/*

clean:
	rm -rf dist
	rm -rf build
