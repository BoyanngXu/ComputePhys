from PIL import Image
import os
n = 2

# 定义图片所在的文件夹路径
folder_path = r"G:\Software\codeFormer-GUI-20221230\_internal\test_img"

# 获取文件夹中所有jpg图片的文件名
file_names = [f for f in os.listdir(folder_path) if f.endswith('.jpg')] + [f for f in os.listdir(folder_path) if f.endswith('.png')]

# 遍历所有图片，并对尺寸进行放大
for file_name in file_names:
    # 打开图片
    image_path = os.path.join(folder_path, file_name)
    with Image.open(image_path) as im:
        # 将RGBA转换为RGB
        if im.mode == "RGBA":
            im = im.convert("RGB")
        # 获取原始尺寸
        width, height = im.size
        # 放大尺寸
        new_width = int(width * n)
        new_height = int(height * n)
        # 调整尺寸
        im = im.resize((new_width, new_height))
        # 保存修改后的图片
        new_file_name = os.path.splitext(file_name)[0] + "_resized.jpg"
        new_file_path = os.path.join(folder_path, new_file_name)
        im.save(new_file_path)

# By ChatGPT