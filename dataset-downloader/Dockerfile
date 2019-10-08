FROM python:3.7-alpine

# RUN apt-get update -y

# RUN apt-get install python3-pip -y

RUN pip3 install urllib3==1.25.3

ENV APP_HOME /app

COPY src $APP_HOME

WORKDIR $APP_HOME

ENTRYPOINT ["python3"]
CMD ["task.py"]