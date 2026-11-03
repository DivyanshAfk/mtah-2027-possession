"""Print the versions of the packages the project needs (a quick environment check). Run: python src/check_env.py"""
import sklearn, numpy, pandas
print("sklearn", sklearn.__version__, "numpy", numpy.__version__, "pandas", pandas.__version__)
