import os.path
import re
import json

data = """
STADIUM_CamRot  0 P=-1.200000 Y=67.400001 R=0.000000
STADIUM_CamPos  0 X=-4417.165 Y=-3333.712 Z=425.752
STADIUM_CamFoc  0 37.497356

STADIUM_ViewMatrix  0 [-0.92321 0.00804807 0.384211 0] [0.384295 0.0193343 0.923008 0] [-2.86229e-16 0.999781 -0.0209424 0] [-2796.84 -325.655 4783.08 1] 
STADIUM_Projection  0 [5.23756 0 0 0] [0 9.31122 0 0] [0 0 0 1] [0 0 10 0] 
STADIUM_ViewProjection  0 [-4.83537 0.0749374 0 0.384211] [2.01277 0.180026 0 0.923008] [-1.49914e-15 9.30918 0 -0.0209424] [-14648.6 -3032.24 10 4783.08] 
"""

def get_data(data):
    # Define regular expressions for different patterns
    pattern_cam_rot = r'STADIUM_CamRot\s+(\d+)\s+P=(-?\d+\.\d+)\s+Y=(-?\d+\.\d+)\s+R=(-?\d+\.\d+)'
    pattern_cam_pos = r'STADIUM_CamPos\s+(\d+)\s+X=(-?\d+\.\d+)\s+Y=(-?\d+\.\d+)\s+Z=(-?\d+\.\d+)'
    pattern_cam_foc = r'STADIUM_CamFoc\s+(\d+)\s+(-?\d+\.\d+)'
    pattern_matrix = r'STADIUM_(ViewMatrix|Projection|ViewProjection)\s+(\d+)\s+\[([\d\s.e-]+)\]\s+\[([\d\s.e-]+)\]\s+\[([\d\s.e-]+)\]\s+\[([\d\s.e-]+)\]'

    matches = re.findall(pattern_cam_rot, data)
    cam_rot_data = [{"index": match[0], "P": match[1], "Y": match[2], "R": match[3]} for match in matches]

    matches = re.findall(pattern_cam_pos, data)
    cam_pos_data = [{"index": match[0], "X": match[1], "Y": match[2], "Z": match[3]} for match in matches]

    matches = re.findall(pattern_cam_foc, data)
    cam_foc_data = [{"index": match[0], "value": match[1]} for match in matches]

    matches = re.findall(pattern_matrix, data)
    matrix_data = [{"type": match[0], "index": match[1], "row1": list(map(float, match[2].split())),
                    "row2": list(map(float, match[3].split())), "row3": list(map(float, match[4].split())),
                    "row4": list(map(float, match[5].split()))} for match in matches]

    matrix_dict = {}
    for d in matrix_data:
        matrix_dict[d["type"]] = [d["row1"], d["row2"], d["row3"], d["row4"]]
    # Combine all the extracted data
    output_data = {
        "STADIUM_CamRot": [float(cam_rot_data[0][key]) for key in "PYR"],
        "STADIUM_CamPos": [float(cam_pos_data[0][key]) for key in "XYZ"],
        "STADIUM_CamFoc": float(cam_foc_data[0]["value"]),
        "STADIUM_Matrix": matrix_dict
    }
    return output_data

# output_data=get_data(data)
# # Convert to JSON
# output_json = json.dumps(output_data, indent=4)
# print(output_json)


def load_cameras(path,dst_dir):
    with open(path, 'r') as f:
        data=f.readlines()
    data=list(filter(lambda x:x.strip(),data))
    cameras={}
    projs={}
    for i in range(4):
        camera_data=data[i*6:(i+1)*6]
        camera_data=get_data("\n".join(camera_data))
        cameras[str(i)]=camera_data
        projs[str(i)]=camera_data["STADIUM_Matrix"]["ViewProjection"]
    with open(os.path.join(dst_dir,"cameras.json"),"w") as f:
        json.dump(cameras,f,indent=2)
    with open(os.path.join(dst_dir,"proj.json"),"w") as f:
        json.dump(projs,f,indent=2)
    # print(cameras)


load_cameras("/home/tww/Datasets/ue/val/camera.txt","/home/tww/Datasets/ue/val")


