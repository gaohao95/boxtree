import pyopencl as cl
import numpy as np
import boxtree
from mpi4py import MPI
from sumpy.kernel import LaplaceKernel
from sumpy.expansion import DefaultExpansionFactory
from sumpy.fmm import SumpyTreeIndependentDataForWrangler, SumpyExpansionWrangler
from functools import partial


def fmm_level_to_order(base_kernel, kernel_arg_set, tree, level):
    return max(level, 3)


def main():
    dims = 3
    nsources = 10000
    ntargets = 10000
    dtype = np.float64

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()

    from boxtree.traversal import FMMTraversalBuilder
    traversal_builder = FMMTraversalBuilder(ctx, well_sep_is_n_away=2)

    kernel = LaplaceKernel(dims)
    expansion_factory = DefaultExpansionFactory()
    local_expansion_factory = expansion_factory.get_local_expansion_class(kernel)
    local_expansion_factory = partial(local_expansion_factory, kernel)
    multipole_expansion_factory = \
        expansion_factory.get_multipole_expansion_class(kernel)
    multipole_expansion_factory = partial(multipole_expansion_factory, kernel)

    tree_indep = SumpyTreeIndependentDataForWrangler(
        ctx, multipole_expansion_factory, local_expansion_factory, [kernel])

    def wrangler_factory(local_traversal, global_traversal):
        from boxtree.distributed.calculation import DistributedSumpyExpansionWrangler
        return DistributedSumpyExpansionWrangler(
            ctx, comm, tree_indep, local_traversal, global_traversal, dtype,
            fmm_level_to_order,
            communicate_mpoles_via_allreduce=True)

    global_tree_dev = None
    sources_weights = cl.array.empty(queue, 0, dtype=dtype)

    if mpi_rank == 0:
        # Generate random particles and source weights
        from boxtree.tools import make_normal_particle_array as p_normal
        sources = p_normal(queue, nsources, dims, dtype, seed=15)
        targets = p_normal(queue, ntargets, dims, dtype, seed=18)

        from pyopencl.clrandom import PhiloxGenerator
        rng = PhiloxGenerator(queue.context, seed=20)
        sources_weights = rng.uniform(queue, nsources, dtype=np.float64)

        rng = PhiloxGenerator(queue.context, seed=22)
        target_radii = rng.uniform(
            queue, ntargets, a=0, b=0.05, dtype=np.float64).get()

        # Build the tree and interaction lists
        from boxtree import TreeBuilder
        tb = TreeBuilder(ctx)
        global_tree_dev, _ = tb(
            queue, sources, targets=targets, target_radii=target_radii,
            stick_out_factor=0.25, max_particles_in_box=30, debug=True)

        global_trav_dev, _ = traversal_builder(queue, global_tree_dev, debug=True)

        wrangler = SumpyExpansionWrangler(tree_indep, global_trav_dev, dtype,
                                          fmm_level_to_order)

        shmem_potential = boxtree.fmm.drive_fmm(wrangler, [sources_weights])

    from boxtree.distributed import DistributedFMMRunner
    distribued_fmm_info = DistributedFMMRunner(
        queue, global_tree_dev, traversal_builder, wrangler_factory)

    distributed_potential = distribued_fmm_info.drive_dfmm([sources_weights])

    if mpi_rank == 0:
        assert(shmem_potential.shape == (1,))
        assert(distributed_potential.shape == (1,))

        shmem_potential = shmem_potential[0].get()
        distributed_potential = distributed_potential[0].get()

        print(np.linalg.norm(distributed_potential - shmem_potential, ord=np.inf))


if __name__ == "__main__":
    main()
