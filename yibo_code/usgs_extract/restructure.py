import os
import sys
import pandas as pd
import shutil

if __name__ == "__main__":
    path = sys.argv[1]
    files = pd.read_csv(path)
    for index, row in files.iterrows():
        try:
            source = row["file_path"]
            file_name = os.path.basename(source)

            # Destination file path
            destination = f"/home/waves/data/usgs_extract/UpdatedDataJan/{row["id"]}/{file_name}"
            # Move the file
            if not os.path.exists(f"/home/waves/data/usgs_extract/UpdatedDataJan/{row["id"]}"):
                os.makedirs(f"/home/waves/data/usgs_extract/UpdatedDataJan/{row["id"]}")
            shutil.move(source, destination)

            print("File moved successfully!")
        except Exception as e:
            print(e)