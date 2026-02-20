import math
import os
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard, Partial

DEVICE_TYPE='cuda'

# Return the two factors of `n` that are closest together.
# Returns a tuple (a, b) of these two factors, with a >= b.
def closest_factors(n):
    sqrt = math.isqrt(n)

    for i in range(sqrt, 0, -1):
        if n % i == 0:
            # Return the pair of factors
            return (n // i, i)

def get_rank_from_env():
    if 'PMI_RANK' in os.environ:
        return os.environ['PMI_RANK']
    elif 'PMIX_RANK' in os.environ:
        return os.environ['PMIX_RANK']
    elif 'RANK' in os.environ:
        return os.environ['RANK']
    else:
        raise RuntimeError('Error: neither \'PMI_RANK\' nor \'RANK\' environment variable found. Are you invoking this script using mpirun or torchrun?')

def get_nprocs_from_env():
    if 'PMI_SIZE' in os.environ:
        return os.environ['PMI_SIZE']
    elif 'WORLD_SIZE' in os.environ:
        return os.environ['WORLD_SIZE']
    else:
        raise Exception('Error: neither \'PMI_SIZE\' nor \'WORLD_SIZE\' environment variable found. Are you invoking this script using mpirun or torchrun?')

def materialized_placements(device_mesh, placements):
    new_placements = []
    has_partial = False

    partial_dims = {0, 1}
    for p in placements:
        if isinstance(p, Partial):
            has_partial = True
        elif isinstance(p, Shard):
            partial_dims.remove(p.dim)
        elif isinstance(p, Replicate):
            continue

    if has_partial:
        materialize_dim = partial_dims.pop()

        for p in placements:
            if isinstance(p, Partial):
                new_placements.append(Shard(materialize_dim))
            else:
                new_placements.append(p)
    else:
        new_placements = placements

    return has_partial,new_placements

def two_dimensional_partitioning(process_grid=None, replication_factor=1, mesh_type=DEVICE_TYPE):
    world_size = dist.get_world_size()

    if world_size % replication_factor != 0:
        raise RuntimeError(f"World size {world_size} not divisible by replication factor {replication_factor}")

    processes_per_replica = world_size // replication_factor

    if process_grid == None:
        process_grid = closest_factors(processes_per_replica)

    mesh_structure = torch.arange(world_size).reshape((replication_factor, *process_grid))

    # Remove the outer dimension, since there is no replication.
    if replication_factor == 1:
        mesh_structure = mesh_structure.reshape(mesh_structure.shape[1:])

    if replication_factor == world_size:
        mesh_structure = mesh_structure.flatten()

    mesh = DeviceMesh(mesh_type, mesh_structure)

    if replication_factor == 1:
        return (mesh, [Shard(0), Shard(1)])
    elif replication_factor == world_size:
        return (mesh, [Replicate()])
    else:
        return (mesh, [Replicate(), Shard(0), Shard(1)])

def one_dimensional_partitioning(dimension = "row", replication_factor=1, mesh_type=DEVICE_TYPE):
    if dimension == "row":
        sharding = Shard(0)
    elif dimension == "column":
        sharding = Shard(1)
    else:
        raise RuntimeError(f"dimension {dimension} is not valid for one_dimensional_partitioning")

    world_size = dist.get_world_size()

    if world_size % replication_factor != 0:
        raise RuntimeError(f"World size {world_size} not divisible by replication factor {replication_factor}")

    # Split up world into `replication_factor` groups.
    replica_groups = torch.chunk(torch.arange(world_size), chunks=replication_factor, dim=0)

    mesh_structure = torch.stack(replica_groups)

    if replication_factor == 1 or replication_factor == world_size:
        mesh_structure = mesh_structure.flatten()

    mesh = DeviceMesh(mesh_type, mesh_structure)

    if replication_factor == 1:
        return (mesh, [sharding])
    elif replication_factor == world_size:
        return (mesh, [Replicate()])
    else:
        return (mesh, [Replicate(), sharding])

def row_partitioning(replication_factor=1, mesh_type=DEVICE_TYPE):
    return one_dimensional_partitioning("row", replication_factor, mesh_type)

def column_partitioning(replication_factor=1, mesh_type=DEVICE_TYPE):
    return one_dimensional_partitioning("column", replication_factor, mesh_type)
