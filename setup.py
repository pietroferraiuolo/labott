from setuptools import setup, find_packages

# Read version and other metadata from labott/__version__.py
about = {}
with open("labott/__version__.py") as f:
    exec(f.read(), about)

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name=about["__title__"],
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    description=about["__description__"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url=about["__url__"],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
)
