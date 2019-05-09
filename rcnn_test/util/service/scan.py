import os

from util.service.time_decorator import spend_time


@spend_time
def scan_files(args):
    file_list = []
    image_path = os.path.join(args.image_path)
    for root, dirs, files in os.walk(image_path):
        # scan all files and put it into list
        for f in files:
            ext_list = [ext.lower() for ext in args.exts.split(',')]
            if os.path.splitext(f)[1].lower() in ext_list:
                file_path = os.path.join(root, f).replace(os.path.join(image_path, ""), "", 1)
                file_list.append(file_path)
    return file_list
