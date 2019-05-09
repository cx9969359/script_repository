import os,argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_path",help="path to image and json folders")

def rename_files(files_path):
    for root, dirs, files in os.walk(files_path):
        for f in files:
                d = f
                d = d.replace(" ","_")
                d = d.replace("+","_")
                d = d.replace("-","_")
                d = d.replace("__","_")
                #print(os.path.join(root, d))
                os.rename(os.path.join(root, f),os.path.join(root, d))


if __name__ == '__main__':
    args = parser.parse_args()
    rename_files(args.input_path)
