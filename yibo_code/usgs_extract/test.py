import os

root_json = "/home/waves/data/usgs_extract"

if os.path.isdir(root_json):
    print("Directory Exists")
else:
    print("No Such Directory")