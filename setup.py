from setuptools import setup, find_packages

setup(
    name="mlb_pred",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy", 
        "matplotlib",
        "seaborn",
        "pybaseball",
        "scikit-learn",
        "tqdm",
        "xgboost",
        "lightgbm"
    ],
    python_requires=">=3.10",
)