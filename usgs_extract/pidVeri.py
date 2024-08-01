import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import json

if __name__ == "__main__":

    download_folder = "/home/waves/data/usgs_extract/Cali_pdf_extended"
    classurl = "usa-link Document"
    url_csv = "/home/waves/data/usgs_extract/csvs/cali1980.csv"
    data = pd.read_csv(url_csv)
    downloaded_pdfs = os.listdir(download_folder)

    obtainedPID = []
    for pdf in downloaded_pdfs:
        obtainedPID.append(int(pdf[:-4]))
    print(obtainedPID)
    pdfID = list(data["Publication ID"])
    print(pdfID)

    nanPid = []
    for id in pdfID:
        if id in obtainedPID:
            print(f"{id} downloaded sucessfully")
        else:
            print(f"{id} not downloaded")
            nanPid.append(id)
    with open("file.json", 'w') as f:
        json.dump(nanPid, f, indent=2)

    print(len(obtainedPID)/len(pdfID)) 
    