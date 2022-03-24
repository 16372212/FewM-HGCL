import os


def get_all_files_by_dir(path: str):
    """read fimes from this directory"""
    L = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.join(root, file))
    return L
