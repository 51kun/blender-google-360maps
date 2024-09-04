import fitz  # PyMuPDF
import os
from PIL import Image, ImageDraw
#视频教程：https://space.bilibili.com/627908600
# 安装库的指令：python -m pip install PyMuPDF Pillow scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
# 获取当前脚本所在目录
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
# 去掉最后的文件名部分
script_dir = os.path.dirname(script_dir)

# 定义 PDF 文件目录
pdf_dir = os.path.join(script_dir, 'pdf')  # PDF 文件目录
# 检查 PDF 目录是否存在
if not os.path.isdir(pdf_dir):
    raise FileNotFoundError(f"指定的路径 '{pdf_dir}' 不存在。请检查目录是否正确。")

def draw_center_lines(image_path):
    """在图片上绘制中线，并保存覆盖原图"""
    # 打开图片
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # 绘制横线
        draw.line((0, height // 2, width, height // 2), fill="white", width=1)
        # 绘制竖线
        draw.line((width // 2, 0, width // 2, height), fill="white", width=1)

        # 保存覆盖原图
        img.save(image_path)

# 遍历指定目录下的所有 PDF 文件
for filename in os.listdir(pdf_dir):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, filename)
        pdf_document = fitz.open(pdf_path)

        # 只处理第一页
        page = pdf_document.load_page(0)
        image_list = page.get_images(full=True)

        if image_list:
            # 只提取第一页的第一张图片
            img_index = 0
            image = image_list[img_index]
            xref = image[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # 保存图片到 PDF 文件所在的目录
            image_filename = f"{os.path.splitext(filename)[0]}.png"
            image_path = os.path.join(pdf_dir, image_filename)
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            
            # 绘制中线并覆盖原图
            draw_center_lines(image_path)
            print(f"Saved {image_filename}")

        pdf_document.close()
