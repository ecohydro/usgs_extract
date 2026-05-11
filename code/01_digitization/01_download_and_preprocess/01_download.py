import argparse
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def get_pdf_url(pub_url, class_name="usa-link Document"):
    try:
        response = requests.get(pub_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        base_url = urljoin(pub_url, "/")
        return [urljoin(base_url, a["href"]) for a in soup.find_all("a", class_=class_name, href=True)]
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {pub_url}: {e}")
        return []


def download_pdf(url, output_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download USGS PDFs from a publication list CSV.")
    parser.add_argument("--input-csv", required=True, help="CSV with 'URL' and 'Publication ID' columns")
    parser.add_argument("--output-dir", required=True, help="Directory to save downloaded PDFs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = pd.read_csv(args.input_csv)

    for _, row in data.iterrows():
        pub_id = row["Publication ID"]
        out_path = os.path.join(args.output_dir, f"{pub_id}.pdf")
        if os.path.exists(out_path):
            print(f"Skipping {pub_id} (already downloaded)")
            continue
        for url in get_pdf_url(row["URL"]):
            download_pdf(url, out_path)
