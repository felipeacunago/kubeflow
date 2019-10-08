FROM python:3.6.8-slim
MAINTAINER Felipe Acuña <felipe.acunago@gmail.com>

# RUN apk add --no-cache --virtual .build-deps gcc musl-dev \
#     && pip install cython \
#     && apk del .build-deps

# RUN pip install numpy
# RUN pip install matplotlib
# RUN pip install pystan
# RUN pip install fbprophet
# RUN pip install scikit-learn
# RUN pip install seaborn

RUN apt-get -y update  && apt-get install -y \
  python3-dev \
  apt-utils \
  python-dev \
  build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade setuptools

RUN pip install numpy
RUN pip install pystan
RUN pip install fbprophet
RUN pip install google-cloud-storage
RUN pip install gcsfs
RUN pip install retry==0.9.2

ENV APP_HOME /app

COPY src $APP_HOME

WORKDIR $APP_HOME

ENTRYPOINT ["python3"]
CMD ["train.py"]