import numpy as np
import  json


def write_proj_json():
    path="/home/tww/Projects/4d_association-windows/data/shelf/proj.txt"
    output_path="/home/tww/Projects/4d_association-windows/data/shelf/proj.json"
    with open(path,'r') as f:
        a = dict()
        for i in range(6):
            frame_idx=f.readline().strip()
            lines=[]
            for j in range(3):
                line=f.readline().strip().split()
                line=[float(x) for x in line]
                lines.append(line)
            a[frame_idx]=lines
    with open(output_path,'w') as f:
        json.dump(a,f)
write_proj_json()
