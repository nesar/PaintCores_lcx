from mpi4py import MPI
import os
import multiprocessing

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = MPI.Get_processor_name()

print(f"Rank {rank}/{size} on {hostname}: os.cpu_count() = {os.cpu_count()}, multiprocessing.cpu_count() = {multiprocessing.cpu_count()}")
