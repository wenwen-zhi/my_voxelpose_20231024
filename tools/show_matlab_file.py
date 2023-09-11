import numpy as np


def show_tree(data):
    if isinstance(data,list):
        return "<List>"+show_tree(data[0])
    return "element"



def show(path):
    from pymatreader import read_mat
    mat=read_mat(path)["actor3D"]
    mat=np.array(mat)
    print(mat.shape,)
    print(mat)
    # print(show_tree(mat))
def main():
    path="/home/tww/Downloads/Projects/Shelf/actorsGT.mat"
    show(path)

if __name__ == '__main__':
    main()