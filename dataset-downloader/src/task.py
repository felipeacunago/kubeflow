import urllib.request
import argparse
from pathlib import Path

# parser arguments
# python task.py --download-url http://i3.ytimg.com/vi/J---aiyznGQ/mqdefault.jpg --local-path C:/Users/Felipe/Downloads/cat.jpg
parser = argparse.ArgumentParser(description='Docker that downloads data from an URL')
parser.add_argument('--download-url', type=str, help='URL of the file to download')
parser.add_argument('--local-path', type=str, help='Local path where the file will be downloaded')
parser.add_argument('--output-path-file', type=str, help='Path of the local file where the Output 1 URL')
args = parser.parse_args()

# local variables
url = args.download_url
local_path = args.local_path

# main process
print('Beginning file download with urllib2...')
urllib.request.urlretrieve(url, local_path)
print('File downloaded successfully...')