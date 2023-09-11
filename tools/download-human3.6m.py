import os
import wget
from concurrent.futures import ThreadPoolExecutor
output_dir="/home/tww/Datasets/human3.6m"
os.makedirs(output_dir,exist_ok=True)
# def progress(current_size,total_size,width):
#     blocks=int(current_size*100/total_size)
#     msg="%s"%blocks+"["+"■"*blocks+" "*(100-blocks)+"] "+"width=%s"%width
#     print(msg,end="",flush=True)
executor = ThreadPoolExecutor(max_workers=5)

for i in range(1,12):
    url="http://visiondata.cis.upenn.edu/volumetric/h36m/S%s.tar"%(i)
    output_path=os.path.join(output_dir,os.path.basename(url))
    os.system("wget -P %s %s"%(output_path,url))

    a = executor.submit(lambda :os.system("wget -P %s %s"%(output_path,url)))
    # wget.download(url,output_path,progress)
