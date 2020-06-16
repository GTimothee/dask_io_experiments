import sys, json, argparse

DEBUG = False

def get_theta(buffers_volumes, buffer_index, _3d_index, O, B):
    T = list()
    Cs = list()
    for dim in range(len(buffers_volumes[buffer_index].p1)):
        if B[dim] < O[dim]:
            C = 0 
        else:            
            C = ((_3d_index[dim]+1) * B[dim]) % O[dim]
            # print(f'{((_3d_index[dim]+1) * B[dim])}mod{O[dim]} = {C}')
            if C == 0 and B[dim] != O[dim]:  # particular case 
                C = O[dim]

        if C < 0:
            raise ValueError("modulo should not return negative value")

        Cs.append(C)
        T.append(B[dim] - C)   
    
    if DEBUG: 
        print(f'\nProcessing buffer {buffer_index}')
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

    cases = [
        {
            "type": 2,
            "R": [3900,3000,3500],
            "I": [780,600,700],
            "O": [650,500,500],
            "B": [390,600,700],
            "volumestokeep": [1,2,3]
        }, {
            "type": 2,
            "R": [3900,3000,3500],
            "I": [390,300,350],
            "O": [650,500,700],
            "B": [390,600,700],
            "volumestokeep": [1,2,3]
        }, {
            "type": 2,
            "R": [3900,3000,3500],
            "I": [390,300,350],
            "O": [325,250,250],
            "B": [195,300,350],
            "volumestokeep": [1,2,3]
        },
        {
            "type": 3,
            "R": [3900,3000,3500],
            "I": [780,600,700],
            "O": [780,3000,700],
            "B": [390,3000,700],
            "volumestokeep": [1,2,3]
        },
        {
            "type": 3,
            "R": [3900,3000,3500],
            "I": [780,600,700],
            "O": [780,3000,3500],
            "B": [390,3000,3500],
            "volumestokeep": [1,2,3]
        },
        {
            "type": 3,
            "R": [3900,3000,3500],
            "I": [780,600,700],
            "O": [3900,3000,3500],
            "B": [390,3000,3500],
            "volumestokeep": [1,2,3]
        },
        {
            "type": 3,
            "R": [3900,3000,3500],
            "I": [3900,3000,3500],
            "O": [780,600,700],
            "B": [390,3000,3500],
            "volumestokeep": [1,2,3]
        }
    ]

    import dask_io
    from dask_io.optimizer.utils.utils import numeric_to_3d_pos
    from dask_io.optimizer.cases.resplit_utils import get_named_volumes, get_blocks_shape

    import logging
    import logging.config
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
    })

    for case in cases:
        _type, R, O, I, B, volumestokeep = int(case["type"]), tuple(case["R"]), tuple(case["O"]), tuple(case["I"]), tuple(case["B"]), case["volumestokeep"]
        print(f'Current run ------ \nType: {_type}\nR: {R},\nO: {O},\nI: {I}\nvolumestokeep: {volumestokeep}')

        buffers_partition = get_blocks_shape(R, B)
        buffers_volumes = get_named_volumes(buffers_partition, B)

        # find omega and theta max
        omega_max = [0,0,0]
        T_max = [0,0,0]
        for buffer_index in buffers_volumes.keys():
            _3d_index = numeric_to_3d_pos(buffer_index, buffers_partition, order='F')
            T, Cs = get_theta(buffers_volumes, buffer_index, _3d_index, O, B)

            for i in range(3):
                if Cs[i] > omega_max[i]:
                    omega_max[i] = Cs[i]
                if T[i] > T_max[i]:
                    T_max[i] = T[i]

        print("Omega max: ", omega_max)

        nb_bytes_per_voxel = 2
        buffersize = B[0]*B[1]*B[2]
        n = R[2]/B[2]
        N = R[1]/B[1] * R[2]/B[2]

        i, j, k = 0, 1, 2
        F1 = omega_max[k] * min(B[j],T_max[j]) * min(B[i],T_max[i])
        F2 = T_max[k] * max(0, min(B[j] - T_max[j] , omega_max[j])) * min(B[i], T_max[i])
        F3 = omega_max[k] * max(0, min(B[j] - T_max[j] , omega_max[j] )) * min(B[i] , T_max[i] )
        F4 = T_max[k] * T_max[j] * max(0, min(B[i] - T_max[i] , omega_max[i] ))
        F5 = omega_max[k] * T_max[j] * max(0, min(B[i] - T_max[i] , omega_max[i] ))
        F6 = T_max[k] * omega_max[1] * max(0, min(B[i] - T_max[i] , omega_max[i] ))
        F7 = omega_max[k] * omega_max[j] * max(0, min(B[i] - T_max[i] , omega_max[i] ))

        print('F1:', F1)
        print('F2:', F2)
        print('F3:', F3)
        print('F4:', F4)
        print('F5:', F5)
        print('F6:', F6)
        print('F7:', F7)

        print('buffer size: ', buffersize*nb_bytes_per_voxel/1000000000, "GB")
        max_mem = (F1 + n*(F2 + F3) + N*(F4 + F5 + F6 + F7) + buffersize) * nb_bytes_per_voxel
        print("max_mem: ", max_mem/1000000000, "GB")