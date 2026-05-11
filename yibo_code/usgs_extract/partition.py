import os, shutil
import json
root = "/home/waves/data/usgs_extract/ReductJson"
json_dirs = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]

doc_paths = []

for id in json_dirs:
    doc_paths.append(os.path.join(root, id))


count = 0

for doc in doc_paths:
    jsons = os.listdir(doc)
    for j in jsons:
        path = os.path.join(doc, j)
        with open(path) as json_file:
            try:
                parsed = json.load(json_file)['result']['chunks']
            except Exception as e:
                print(e)
                continue

        for content in parsed:
            for block in content["blocks"]:
                if block["type"] == "Table":
                    count += 1
    print(count)

print(count)
                    