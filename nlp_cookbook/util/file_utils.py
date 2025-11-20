def read_text_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()
