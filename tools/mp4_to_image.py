import glob
import os

from cv2 import VideoCapture
from cv2 import imwrite
import cv2


# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型
def save_image(image, addr, num):
    address = os.path.join(addr, str(num) + ".jpg")
    imwrite(address, image)


def convert_video_to_image(video_path, out_dir):
    '''

    '''
    os.makedirs(out_dir, exist_ok=True)

    is_all_frame = True  # 是否取所有的帧
    sta_frame = 1  # 开始帧
    end_frame = 1e+10  # 结束帧

    ######
    time_interval = 1  # 时间间隔

    # 读取视频文件
    videoCapture = VideoCapture(video_path)

    # 读帧
    success, frame = videoCapture.read()
    print(success)

    i = 0
    j = 0
    if is_all_frame:
        time_interval = 1

    while success:
        i = i + 1
        if (i % time_interval == 0):
            if is_all_frame == False:
                if i >= sta_frame and i <= end_frame:
                    j = j + 1
                    print('save frame:', i, out_dir)
                    save_image(frame, out_dir, j)
                elif i > end_frame:
                    break
            else:
                j = j + 1
                print('save frame:', i,out_dir)
                save_image(frame, out_dir, j)

        success, frame = videoCapture.read()


def convert(
        video_dir, out_dir
):
    video_file_list = glob.glob(video_dir + "/*.mp4")  # 视频文件名
    for video_path in video_file_list:
        filename = os.path.basename(video_path)
        video_name = filename.split(".")[0]
        video_out_dir = os.path.join(out_dir, video_name)
        print("正在将%s转换到%s" % (video_path, video_out_dir))
        convert_video_to_image(video_path, video_out_dir)
    print("转换完成")


if __name__ == '__main__':
    convert(
            "/home/tww/Projects/4d_association-windows/data/shelf/video",
        "/home/tww/Projects/4d_association-windows/data/shelf/images"
        )
