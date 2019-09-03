"""Module for wrapping MPI functions (future use)"""

def nprocs():
    """Gets number of processes or returns 1 if mpi is not installed
    
    Returns
    -------
    int:
        Rank of process or 1 if MPI is not installed
    
    """

    try:
        from mpi4py import MPI
    except ImportError:
        return 0  

    comm = MPI.COMM_WORLD

    
    return comm.Get_size()   

def allgather(value):
    import numpy as np
    try:
        from mpi4py import MPI
    except ImportError:
        return value 

    comm = MPI.COMM_WORLD
    data = value
    data = comm.allgather(data)

    return data
    
def broadcast(array,rank=0):
    import numpy as np
    try:
        from mpi4py import MPI
    except ImportError:
        return array
    comm = MPI.COMM_WORLD
    if isinstance(array,np.ndarray):
            

        data = None
        if get_rank() == rank:
            data = np.copy(array)
        else:
            data = np.zeros_like(array)
        comm.Bcast(data, root=rank)
    else:
        data = comm.bcast(data,root=rank)
    
    return data

def get_rank():
    """Gets rank or returns 0 if mpi is not installed
    
    Returns
    -------
    int:
        Rank of process or 0 if MPI is not installed
    
    """

    try:
        from mpi4py import MPI
    except ImportError:
        return 0
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    return rank


def barrier():
    """Gets rank or returns 0 if mpi is not installed
    
    Returns
    -------
    int:
        Rank of process or 0 if MPI is not installed
    
    """

    try:
        from mpi4py import MPI
    except ImportError:
        return

    comm = MPI.COMM_WORLD
    comm.Barrier()