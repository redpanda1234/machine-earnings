.PHONY: all
all:
	python3 poly_fit.py --cache-data

.PHONY: download
download:
	python3 scraper.py

.PHONY: cached
cached:
	python3 poly_fit.py --cache-data --use-cached-data
