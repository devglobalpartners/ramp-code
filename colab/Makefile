help:
	@echo "- install: Installs required packages by ramp & solaris"
	@echo "- clean: Cleans up pycache & ipynb checkpoints"
    
install:
	pip install -r requirements.txt
	pip install ../solaris --use-feature=in-tree-build
	pip install scikit-fmm

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache