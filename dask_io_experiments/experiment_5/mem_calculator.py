import sys, json, argparse

def get_theta(buffers_volumes, buffer_index, _3d_index, O, B):
    T = list()
    Cs = list()
    for dim in range(len(buffers_volumes[buffer_index].p1)):
        if B[dim] < O[dim]:
            C = 0 
        else:            
            C = ((_3d_index[dim]+1) * B[dim]) % O[dim]
            print(f'{((_3d_index[dim]+1) * B[dim])}mod{O[dim]} = {C}')
            if C == 0 and B[dim] != O[dim]:  # particular case 
                C = O[dim]

        if C < 0:
            raise ValueError("modulo should not return negative value")

        Cs.append(C)
        T.append(B[dim] - C)   
    print(f'C: {Cs}')
    print(f'theta: {T}')
    return T, Cs
    

def get_arguments():
    """ Get arguments from console command.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_filepath', 
        action='store', 
        type=str, 
        help='Path to configuration file containing paths of third parties libraries, projects, data directories, etc. See README for more information.')
    return parser.parse_args()


def custom_imports(paths):
    def isempty(s):
        if s == "":
            return True 
        return False 

    for k, path in paths.items():
        if "lib_" in k and not isempty(path):
            sys.path.insert(0, path)


if __name__ == "__main__":

    args = get_arguments()
    with open(args.config_filepath) as f:
        paths = json.load(f)
        custom_imports(paths)

    import dask_io
    from dask_io.optimizer.utils.utils import numeric_to_3d_pos
    from dask_io.optimizer.cases.resplit_utils import get_named_volumes, get_blocks_shape

    R = (3900,3000,3500)
    O = (300, 250, 250)
    B = Lambda = (390,300,350)

    buffers_partition = get_blocks_shape(R, B)
    buffers_volumes = get_named_volumes(buffers_partition, B)

    max_C = [0,0,0]
    for buffer_index in buffers_volumes.keys():
        print(f'\nProcessing buffer {buffer_index}')
        _3d_index = numeric_to_3d_pos(buffer_index, buffers_partition, order='F')
        T, Cs = get_theta(buffers_volumes, buffer_index, _3d_index, O, B)
        for i in range(3):
            if Cs[i] > max_C[i]:
                max_C[i] = Cs[i]

    print("Omega max: ", max_C)
    nb_bytes_per_voxel = 2
    max_mem = (B[0]*B[1]*max_C[2] + B[0]*B[1]*B[2])*nb_bytes_per_voxel
    print("max_mem: ", max_mem, "bytes")