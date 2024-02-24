import os

def mkDir(folder: str):
    """ mkdir -p folder """
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except FileExistsError:
            pass
