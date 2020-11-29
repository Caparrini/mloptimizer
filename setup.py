import setuptools
from pip._internal.req import parse_requirements

with open("README.md", "r") as fh:
    long_description = fh.read()


# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("requirements.txt", session=False)

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setuptools.setup(
    name='mloptimizer',
    version='0.3',
    author="Antonio Caparrini",
    author_email="a.caparrini@gmail.com",
    description="Genetic hyper-parameter selection for machine learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Caparrini/mloptimizer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=reqs,
    python_requires='>=3.8',
)
