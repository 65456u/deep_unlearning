import os

def collect_python_files(directory):
    """
    遍历指定目录及其子目录，收集所有以 .py 结尾的Python文件路径。
    """
    python_files = []
    for root, dirs, files in os.walk(directory):
        # 忽略隐藏文件夹
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                python_files.append(filepath)
    return python_files

def write_to_markdown(python_files, output_file):
    """
    将收集到的Python文件内容写入到一个Markdown文件中，
    每个文件的内容都包含在一个代码块中，并附有文件名作为标题。
    """
    with open(output_file, 'w', encoding='utf-8') as md_file:
        for file in python_files:
            # 写入文件名作为二级标题
            md_file.write(f'## {os.path.relpath(file)}\n\n')
            md_file.write('```python\n')
            with open(file, 'r', encoding='utf-8') as py_file:
                code = py_file.read()
                md_file.write(code)
            md_file.write('\n```\n\n')

if __name__ == "__main__":
    current_directory = os.getcwd()
    python_files = collect_python_files(current_directory)
    output_markdown = 'python_code_collection.md'
    write_to_markdown(python_files, output_markdown)
    print(f"所有Python代码已被收集到 {output_markdown}")
