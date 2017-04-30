"""
Just some tools to keep track of progress record data.
"""

class FileWriterStdoutPrinter:

    def __init__(self, file_path):
        self.path = file_path

    def __enter__(self):
        self.fd = open(self.path, 'w')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fd.close()

    def emit(self, text):
        print text
        self.fd.write(text)

    def emit_line(self, text):
        print text
        self.fd.write(text + '\n')
