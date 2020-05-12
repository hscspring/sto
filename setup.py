import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sto",
    version="0.0.1",
    author="Yam",
    author_email="haoshaochun@gmail.com",
    description="Similar Text only run Once.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hscspring/sto",
    packages=setuptools.find_packages(),
    install_requires=[
          'addict',
          'datasketch',
          'DAWG',
          'numpy',
          'pnlp',
          'PyYAML',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)