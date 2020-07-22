import os
import pandas as pd


def getPathDf(path, ext: str):
    path_list = []
    for _path, _, files in os.walk(path):
        for filename in files:
            _ext = os.path.splitext(filename)[-1]

            if _ext.lower() != ext:
                continue

            path_list.append([os.path.splitext(filename)[:-1], os.path.join(_path, filename)])

    return pd.DataFrame(path_list, columns=['filename', 'path'])
