import os, cv2


def show_files(path, all_files):
    '''遍历文件夹，获得要转换的文件名称'''
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            # 拼接文件路径
            all_files.append(path + "/" + file)
    return all_files


def resize_video(path, savepath):
    '''改视频分辨率'''
    cap = cv2.VideoCapture(path)
    success, _ = cap.read()
    # 重新合成的视频在原文件夹，如果需要分开，可以修改file_n
    videowriter = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25, (320, 240))
    while success:
        success, vid1 = cap.read()
        try:
            vid = cv2.resize(vid1, (320, 240), interpolation=cv2.INTER_LINEAR)  # 希望的分辨率大小可以在这里改
            videowriter.write(vid)
        except:
            break


# 所有需要处理的图片的路径 自动遍历文件夹内所有文件（包括子文件） 可以填写路径全称
traversal_file = r"D:\PycharmProjects\proj1\gaitVer2\PD\videos"
# 修改完成后输出的文件夹
output_file = r"D:\PycharmProjects\proj1\gaitVer2\PD\LowVideos"

contents = show_files(traversal_file, [])  # 循环打印show_files函数返回的文件名列表
for content in contents:
    # 遍历修改
    # 判断是否为视频
    if content.endswith('avi') or content.endswith('mp4'):
        print("processing : " + content)
        op = output_file + "/" + os.path.basename(content)
        resize_video(content, op)
