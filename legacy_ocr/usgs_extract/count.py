import os
import sys
if __name__ == "__main__":
    path = sys.argv[1]
    files = os.listdir(path)
    print(len(files))