FROM python:3.7.2-alpine3.9

# RUN apt-get update -y

# RUN apt-get install python3-pip -y

RUN apk add --no-cache \
        --virtual=.build-dependencies \
        g++ gfortran file binutils \
        musl-dev python3-dev openblas-dev && \
    apk add libstdc++ openblas && \
    \
    ln -s locale.h /usr/include/xlocale.h && \
    \
    pip install numpy && \
      pip install pandas==0.25.1 && \
      pip3 install gcsfs && \
      pip install cython && \
      pip install -U scikit-learn && \
    \
    rm -r /root/.cache && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -r '{}' + && \
    find /usr/lib/python3.*/site-packages/ -name '*.so' -print -exec sh -c 'file "{}" | grep -q "not stripped" && strip -s "{}"' \; && \
    \
    rm /usr/include/xlocale.h && \
    \
    apk del .build-dependencies

ENV APP_HOME /app

COPY src $APP_HOME

WORKDIR $APP_HOME

ENTRYPOINT ["python3"]
CMD ["task.py"]