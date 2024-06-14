import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import sys

# data = pd.read_csv("./publications-water-terms1970.csv")
classurl = "usa-link Document"
url_folder = os.path.join(os.getcwd(), "../pdf/publications-water-terms1970.csv")
data = pd.read_csv(url_folder)
urls = data["URL"]
def get_urls_with_class(url, class_name):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        base_url = urljoin(url, '/')
        urls = [urljoin(base_url, a['href']) for a in soup.find_all('a', class_=class_name, href=True)]
        return urls
    except requests.exceptions.RequestException as e:
        print(f'Failed to fetch {url}. Reason: {e}')
        return []

# Function to download a file from a given URL
def download_file(url, folder_path):
    try:
        # Send a HTTP request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Parse the URL to get the path and filename
        parsed_url = urlparse(url)
        name =  parsed_url.path.split('/')
        name = name[-1]
        file_path = os.path.join(folder_path, name)

        # Write the content to a file
        with open(file_path, 'wb') as file:
            file.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f'Failed to download {url}. Reason: {e}')


if __name__ == "__main__":

    download_folder = sys.argv[1]

    # Create the folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Download each file from the list of URLs
    for url in urls:
        pdfurls = get_urls_with_class(url, classurl)
        for urlp in pdfurls:

            download_file(urlp, download_folder)
