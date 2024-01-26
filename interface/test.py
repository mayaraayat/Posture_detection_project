


if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    NbP = comm.Get_size()
    Me = comm.Get_rank()
    print("Hello World from process ",Me,"/",NbP)
    comm.Barrier()