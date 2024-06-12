import time

def process_file(file_path):
    # Misalnya kita akan membuat sebuah pesan sederhana berdasarkan nama file
    file_name = file_path.split('/')[-1]
    custom_message = f'File "{file_name}" telah diproses.'
    time.sleep(5)
    return custom_message
