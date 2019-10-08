FROM tiangolo/uwsgi-nginx-flask:python3.7
MAINTAINER Felipe Acuña <felipe.acunago@gmail.com>

RUN pip install --upgrade setuptools

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/

COPY ./app /app