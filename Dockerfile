FROM python:latest

WORKDIR /mloptimizer

# By copying over requirements first, we make sure that Docker will cache
# our installed requirements rather than reinstall them on every build
COPY requirements.txt /mloptimizer/requirements.txt
RUN pip install -r requirements.txt