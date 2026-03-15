import os
from docx2pdf import convert

# 自动获取当前电脑的桌面路径
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 定义文件名
file_name_docx = "24050137  王挚.docx"
file_name_pdf = "24050137  王挚.pdf"

# 拼接完整的输入和输出路径
input_path = os.path.join(desktop_path, file_name_docx)
output_path = os.path.join(desktop_path, file_name_pdf)


def main():
    # 检查桌面上是否存在该 docx 文件
    if os.path.exists(input_path):
        print(f"成功找到文件: {input_path}")
        print("正在调用 Word 进行转换，请稍候...")

        try:
            # 执行转换
            convert(input_path, output_path)
            print("\n✅ 转换成功！")
            print(f"PDF 文件已保存至桌面上: {output_path}")
        except Exception as e:
            print(f"\n❌ 转换过程中出现错误: {e}")
            print("请检查：1. 电脑是否安装了 Microsoft Word 2. 该文档是否正被其他软件占用")
    else:
        print(f"❌ 找不到文件！请确认【{file_name_docx}】是否已经放在了桌面上。")
        print(f"系统正在查找的路径是: {input_path}")


if __name__ == "__main__":
    main()