import os


def getAllFiles(analysis_root_dir):
    """read fimes from this directory"""
    L = []
    for root, dirs, files in os.walk(analysis_root_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.join(root, file))
    return L
