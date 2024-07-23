import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

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
def download_file(url, folder_path, id):
    try:
        # Send a HTTP request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        file_path = os.path.join(folder_path, f"{pdfID[id]}.pdf")

        # Write the content to a file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"download success at {file_path}")
    except requests.exceptions.RequestException as e:
        print(f'Failed to download {url}. Reason: {e}')


if __name__ == "__main__":

    download_folder = "/home/waves/data/usgs_extract/validation_test/pdfs"
    classurl = "usa-link Document"
    url_csv = "/home/waves/data/usgs_extract/validation_test/valiSet.csv"
    data = pd.read_csv(url_csv)

    urls = data["URL"]
    pdfID = data["Publication ID"]

    # Create the folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Download each file from the list of URLs
    for i, url in enumerate(urls):
        pdfurls = get_urls_with_class(url, classurl)
        for urlp in pdfurls:
            download_file(urlp, download_folder, i)


