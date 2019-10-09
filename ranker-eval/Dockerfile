FROM python:3.7.2-alpine3.9

RUN apk add --no-cache python3-dev libstdc++ && \
    apk add --no-cache g++ && \
    ln -s /usr/include/locale.h /usr/include/xlocale.h && \
    pip3 install numpy && \
    pip3 install pandas && \
    pip3 install google-cloud-storage && \
    pip3 install gcsfs

ENV APP_HOME /app

COPY src $APP_HOME

WORKDIR $APP_HOME

ENTRYPOINT ["python3"]
CMD ["task.py"]