import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='mloptimizer',
    version='0.1',
    scripts=['mloptimizer'],
    author="Antonio Caparrini",
    author_email="a.caparrini@gmail.com",
    description="genetic hyper parameter selection for machine learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Caparrini/mloptimizer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
