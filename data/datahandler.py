#coding:utf-8

def remove_duplicate_lines(input_path: str, output_path: str, encoding: str = "utf-8"):
    """
    高性能文本去重：保留每行唯一，保留首次出现顺序
    :param input_path: 原txt文件路径
    :param output_path: 去重后保存路径
    :param encoding: 文件编码
    """
    seen = set()
    unique_lines = []

    # 逐行读取，内存占用极低，速度最快
    with open(input_path, "r", encoding=encoding) as f:
        for line in f:
            # 去除首尾空白换行（若需要严格按原行原样，去掉strip()）
            raw_line = line.rstrip("\n\r")
            if raw_line not in seen:
                seen.add(raw_line)
                unique_lines.append(line)

    # 一次性写入，减少IO次数，提升性能
    with open(output_path, "w", encoding=encoding) as f:
        f.writelines(unique_lines)

    print(f"去重完成！")
    print(f"原始行数：{len(seen) + sum(1 for _ in open(input_path, encoding=encoding)) - len(unique_lines)}")
    print(f"去重后行数：{len(unique_lines)}")
    print(f"重复行数已剔除：{sum(1 for _ in open(input_path, encoding=encoding)) - len(unique_lines)}")


# ========== 在这里修改你的文件路径 ==========
if __name__ == "__main__":
    # 原文件路径
    input_file = "entities.txt"
    # 去重后保存的新文件路径
    output_file = "entities_去重后.txt"
    remove_duplicate_lines(input_file, output_file)