
def get_rank():
    try:
        from mpi4py import MPI
    except ImportError:
        return 0
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    return rank
    