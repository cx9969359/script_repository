import hashlib
import time


def work():
    with open('F:\\tif_images\\thyroid\\more.tif', 'rb') as f:
        md5_obj = hashlib.md5()
        while True:
            d = f.read(102400)
            if not d:
                break
            md5_obj.update(d)
        hash_code = md5_obj.hexdigest()
        f.close()
        md5 = str(hash_code).lower()
        print(md5)


if __name__ == '__main__':
    start = time.time()
    work()
    print(time.time() - start)
