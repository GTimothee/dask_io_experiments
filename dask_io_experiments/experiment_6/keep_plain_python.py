

def get_buffers():
    pass 


def read(buffer, input_dir):
    pass


def split_data(data):
    pass


def keep(data_to_keep, cache):
    pass


def write(data_to_write, output_dir):
    pass


if __name__ == "__main__":

    buffers = get_buffers()
    cache = dict()
    for buffer in buffers:
        data = read(buffer, input_dir) 
        data_to_write, data_to_keep = split_data(data)
        keep(data_to_keep, cache)
        write(data_to_write, output_dir)