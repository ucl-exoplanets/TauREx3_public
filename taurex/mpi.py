"""Module for wrapping MPI functions (future use)"""
from functools import lru_cache
from functools import wraps
import numpy as np


def convert_op(operation):
    from mpi4py import MPI
    if operation.lower() == 'sum':
        return MPI.SUM
    else:
        raise NotImplementedError

@lru_cache(maxsize=10)
def shared_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)



@lru_cache(maxsize=2)
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
        return 1

    comm = MPI.COMM_WORLD

    return comm.Get_size()


def allgather(value):
    try:
        from mpi4py import MPI
    except ImportError:
        return [value]

    comm = MPI.COMM_WORLD
    data = value
    data = comm.allgather(data)

    return data

def allreduce(value, op):
    try:
        from mpi4py import MPI
    except ImportError:
        return value

    comm = MPI.COMM_WORLD
    data = value
    data = comm.allreduce(value, op=convert_op(op))

    return data


def broadcast(array, rank=0):
    import numpy as np
    try:
        from mpi4py import MPI
    except ImportError:
        return array
    comm = MPI.COMM_WORLD
    if isinstance(array, np.ndarray):

        data = None
        if get_rank() == rank:
            data = np.copy(array)
        else:
            data = np.zeros_like(array)
        comm.Bcast(data, root=rank)
    else:

        data = comm.bcast(array, root=rank)

    return data



@lru_cache(maxsize=2)
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


def only_master_rank(f):
    """
    A decorator to ensure only the master
    MPI rank can run it
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return f(*args, **kwargs)
    return wrapper


@lru_cache(maxsize=10)
def shared_rank():
    return shared_comm().Get_rank()


def allocate_as_shared(arr, logger=None, force_shared=False):
    try:
        from mpi4py import MPI
    except ImportError:
        return arr
    from taurex.cache import GlobalCache
    if GlobalCache()['mpi_use_shared'] or force_shared:
        if logger is not None:
            logger.info('Moving to shared memory')
        comm = shared_comm()
        nbytes = arr.size*arr.itemsize

        window = MPI.Win.Allocate_shared(nbytes, arr.itemsize, comm=comm)
        buf, itemsize = window.Shared_query(0)
        if itemsize != arr.itemsize:
            raise Exception(f'Shared memory size {itemsize} != array itemsize {arr.itemsize}')

        shared_array = np.ndarray(buffer=buf, dtype='d', shape=arr.shape)

        if shared_rank() == 0:
            shared_array[...] = arr[...]

        comm.Barrier()

        return shared_array
    else:
        return arr
