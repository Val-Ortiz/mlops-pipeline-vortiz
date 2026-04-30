install:
	pip install -r requirements.txt

train:
	python src/train.py

test:
	python src/validate.py

lint:
	pip install flake8 && flake8 src/ --max-line-length=120 --ignore=E501
