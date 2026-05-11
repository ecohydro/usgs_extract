import os
import sys
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

key = key = os.getenv('API_KEY')
if __name__ == "__main__":
    root = sys.argv[1]
    json_root = "/home/waves/data/usgs_extract/ReductJson"
    docs = os.listdir(root)

    # Iterate through the files in root directory
    for doc in docs:

        # create the subdirectory paths for each document
        doc_path = os.path.join(root, doc)
        pages = os.listdir(doc_path)
        json_path = os.path.join(json_root, doc)

        # check to see if the directoy already exists
        if not os.path.exists(json_path):
            os.makedirs(json_path)

        # Create the path to each page
        for page in pages:
            page_path = os.path.join(doc_path, page)
            json_page = page[:-4] + ".json"
            if os.path.exists(os.path.join(json_path, json_page)):
                print(os.path.join(json_path, json_page))
                continue
            else:
                try:
                    upload_form = requests.post("https://platform.reducto.ai/upload",
                                    headers={"Authorization": f"Bearer {key}"}).json()
                    requests.put(upload_form["presigned_url"], data=open(page_path, "rb"))
                    response = requests.post("https://platform.reducto.ai/parse",
                                json={"document_url": upload_form["file_id"]},
                                headers={"Authorization": f"Bearer {key}"})
                    json_object = json.loads(response.text)
                    print(os.path.join(json_path, json_page))
                    with open(os.path.join(json_path, json_page), 'w') as f:
                        json.dump(json_object, f, indent=4)
                except Exception as e:
                    print(e)