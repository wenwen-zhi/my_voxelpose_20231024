import json

from pymatreader import read_mat

filename="/home/tww/Datasets/CampusSeq1/CampusSeq1/CampusSeq1/actorsGT.mat"
data = read_mat(filename)

# print(data)
x=data["actor3D"]
print(x[0][0].shape)
# P_list=data["P"]
# output={}
# for i,P in enumerate(P_list):
#     P=P.tolist()
#     output[i]=P
# with open("proj_campus.json","w") as f:
#     json.dump(output,f)
#
# # with open("")