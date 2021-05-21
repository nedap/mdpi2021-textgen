lint:
	pylint --disable=R,C textgen

test:
	pytest --doctest-modules --cov-report html --cov=textgen textgen/ tests/
