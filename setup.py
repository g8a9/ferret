#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "transformers",
    "datasets",
    "sentencepiece",
    "captum",
    "einops",
    "shap",
    "seaborn",
    "matplotlib",
    "numpy",
    "pandas",
    "tqdm",
    "scikit-image",
    "opencv-python",
    "lime",
    "joblib",
    "pytreebank",
]

test_requirements = []

setup(
    author="Giuseppe Attanasio",
    author_email="giuseppeattanasio6@gmail.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    description="A python package for benchmarking interpretability approaches.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,  # + "\n\n" + history,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="ferret",
    name="ferret-xai",
    packages=find_packages(include=["ferret", "ferret.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/g8a9/ferret",
    version="0.3.4",
    zip_safe=False,
)
