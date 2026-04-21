.PHONY: all data notebook clean verify

all: data notebook

data:
	python data/generate_data.py

notebook:
	jupyter nbconvert --to notebook --execute notebook.ipynb \
	  --output notebook_executed.ipynb

verify:
	python -c "import hashlib; \
	  [print(f'{f}: {hashlib.sha256(open(f, \"rb\").read()).hexdigest()[:16]}') \
	   for f in ['data/customers.csv', 'data/subscriptions.csv', \
	             'data/churn_reasons.csv']]"

clean:
	rm -rf data/*.csv outputs/figures/*.png notebook_executed.ipynb
