import setuptools
import os
try:
    # pip >=20
    from pip._internal.network.session import PipSession
    from pip._internal.req import parse_requirements
except ImportError:
    try:
        # 10.0.0 <= pip <= 19.3.1
        from pip._internal.download import PipSession
        from pip._internal.req import parse_requirements
    except ImportError:
        # pip <= 9.0.3
        from pip.download import PipSession
        from pip.req import parse_requirements
with open("README.md", "r") as fh:
    long_description = fh.read()

print(os.listdir())
print(os.getcwd())
import pathlib
print(pathlib.Path(__file__).parent.resolve())
# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("/Users/paradox/workspace/mloptimizer/requirements.txt", session=False)

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.requirement) for ir in install_reqs]

setuptools.setup(
    name='mloptimizer',
    version='0.5.1',
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
