import hashlib
import time
from openslide import OpenSlide
from io import BytesIO

def work():
    path = 'E:/yn_img/L201903670-20190426113102-004.TIFF'
    slide = OpenSlide(path)
    thumbnail = slide.get_thumbnail((250, 500))
    thumbnail.save('E:/yn_img/a.jpeg',format='JPEG')



if __name__ == '__main__':
    work()
