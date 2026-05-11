import os
import shutil

source_root = '/home/waves/data/usgs_extract/ReductJson'
destination = '/home/waves/data/usgs_extract/metadataResults/json'

# Create the destination folder if it doesn't exist
os.makedirs(destination, exist_ok=True)

# Walk through all subdirectories of source_root
for subdir, dirs, files in os.walk(source_root):
    # Skip the source root itself if needed
    if subdir == source_root:
        continue
    for file in files:
        src_path = os.path.join(subdir, file)
        dst_path = os.path.join(destination, file)

        # Handle name conflicts by renaming
        if os.path.exists(dst_path):
            base, ext = os.path.splitext(file)
            i = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(destination, f"{base}_{i}{ext}")
                i += 1

        # Copy file
        shutil.copy2(src_path, dst_path)

    