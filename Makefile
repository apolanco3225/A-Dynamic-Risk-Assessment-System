install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black src/*.py

lint:
	pylint src/*.py

deploy:
	uvicorn user_app:app 

test:
	python -m pytest -vv 

all: install format lint test deploy 