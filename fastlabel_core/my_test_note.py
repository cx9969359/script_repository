import ast
import os
import random
import random
import threading
import time
import time
from multiprocessing import Process, Queue, current_process, freeze_support
import hashlib


def work():
    with open('F:\\tif_images\\thyroid\\more.tif', 'rb') as f:
        md5_obj = hashlib.md5()
        while True:
            d = f.read(8096)
            if not d:
                break
            md5_obj.update(d)
        hash_code = md5_obj.hexdigest()
        f.close()
        md5 = str(hash_code).lower()
        print(md5)


if __name__ == '__main__':
    work()

# {'hsil':[[x1,y1,x2,y2,conf],[x1,x2],[]]}
