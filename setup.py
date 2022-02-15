import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ptsnet",
    version="0.0.17",
    author="Gerardo Riano",
    author_email="griano@utexas.edu",
    description="Parallel Transient Simulation in Water Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gandresr/PTSNET",
    project_urls={
        "Bug Tracker": "https://github.com/gandresr/PTSNET/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: The Unlicense (Unlicense)",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.6"
)