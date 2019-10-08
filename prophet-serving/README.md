# Prophet serving API

## About this app

This is an API to make request to get data predictions using prophet

It needs three parameters that are given using environment variables:

```
PROJECT_NAME
MODELS_PATH
DICTIONARY_PATH
```


## Docker

You can pull the image directly from the docker-hub:

```
docker pull felipeacunago/prophet-serving:latest
```

or build it using

```
docker build -t myimage .
```

Then you can run it as a regular docker container:

```
docker run -d --name mycontainer -p 80:80 image -v /path/to/credentials:/credentials/ \ 
-e PROJECT_NAME='gcs-project-name' \
-e GOOGLE_APPLICATION_CREDENTIALS='/crendetials/credentialsfile.json' \
-e MODELS_PATH='gcs-model-path' \
-e DICTIONARY_PATH='gcs-dictionary-file-path' \
```


## No docker

(The following instructions apply to Windows command line.)

To run this app first clone repository and then open a terminal to the app folder.


Create and activate a new virtual environment (recommended) by running
the following:

On Windows

```
virtualenv venv 
\venv\scripts\activate
```

Or if using linux

```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

Install the requirements:

```
pip install -r requirements.txt
```

Create an .env file inside app folder with the following structure:
```
PROJECT_NAME=
MODELS_PATH=
DICTIONARY_PATH=
```

Set the environment variable FLASK_APP:

Windows:
```
set FLASK_APP=main.py
```
Linux:
```
export FLASK_APP=main.py
```
Run the app:

```
python -m flask run
```
You can run the app on your browser.

## Extra

This project uses the base image from tiangolo to serve the model using uWSGI + nginx

https://github.com/tiangolo/uwsgi-nginx-flask-docker