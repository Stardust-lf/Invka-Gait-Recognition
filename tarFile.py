import tarfile

def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path = dirs)

if __name__ == "__main__":
    for i in range(1,125):
        num = str(i).zfill(3)
        print('untaring',num)
        untar("{0}.tar.gz".format(num), ".")