import os
from setuptools import setup, find_packages

about = {}
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'aoptics', '__version__.py'), 'r') as _:
    exec(_.read(), about)

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name=about["__title__"],
    version=about["__version__"],
    author=about["__author__"],
    maintainer=about["__maintainer__"],
    author_email=about["__author_email__"],
    description=about["__description__"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url=about["__url__"],
    license=about["__license__"],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
    package_data={'aoptics': ['core/_configurations/*.yaml']},
)
