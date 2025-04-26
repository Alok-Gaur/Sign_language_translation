import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def create_directory(label):
    directory = os.path.join('data', label)
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        for file in os.listdir(directory):
            os.remove(file)
        os.removedirs(directory)
        os.makedirs(directory)