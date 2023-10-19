import logging

# Import modules
import numpy as np
import multiprocessing as mp
from functools import partial
from collections import deque
from joblib import Parallel, delayed

from pyswarms.backend.operators import compute_pbest
from pyswarms.backend.topology import Star
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler, OptionsHandler
from pyswarms.base import SwarmOptimizer
from pyswarms.utils.reporter import Reporter


class GlobalBestPSO(SwarmOptimizer):
    def __init__(
        self,n_particles,dimensions,options,bounds=None,
        oh_strategy=None,bh_strategy="periodic",
        velocity_clamp=None,vh_strategy="unmodified",
        center=1.00,ftol=-np.inf,ftol_iter=1,init_pos=None,
    ):

        super(GlobalBestPSO, self).__init__(
            n_particles=n_particles,dimensions=dimensions,
            options=options,bounds=bounds,velocity_clamp=velocity_clamp,
            center=center,ftol=ftol,ftol_iter=ftol_iter,init_pos=init_pos,
        )

        if oh_strategy is None:
            oh_strategy = {}
        # Initialize logger
        self.rep = Reporter(logger=logging.getLogger(__name__))
        # Initialize the resettable attributes
        self.reset()
        # Initialize the topology
        self.top = Star()
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.oh = OptionsHandler(strategy=oh_strategy)
        self.name = __name__


    def optimize(
        self, objective_func, iters, n_processes=None, verbose=True, **kwargs
    ):
        # Apply verbosity
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        #pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history = deque(maxlen=self.ftol_iter)
        
        if n_processes is None:
            n_processes = 1
        
        with Parallel(n_jobs=n_processes, backend="threading") as parallel:
            for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
                # Compute cost for current position and personal best
                # fmt: off
                self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool=parallel, **kwargs)
                self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
                # Set best_cost_yet_found for ftol
                best_cost_yet_found = self.swarm.best_cost
                self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)
                # Save to history
                hist = self.ToHistory(
                    best_cost=self.swarm.best_cost,
                    mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                    mean_neighbor_cost=self.swarm.best_cost,
                    position=self.swarm.position,
                    velocity=self.swarm.velocity,
                )
                self._populate_history(hist)
                # Verify stop criteria based on the relative acceptable cost ftol
                relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
                delta = (
                    np.abs(self.swarm.best_cost - best_cost_yet_found)
                    < relative_measure
                )
                if i < self.ftol_iter:
                    ftol_history.append(delta)
                else:
                    ftol_history.append(delta)
                    if all(ftol_history):
                        break
                # Perform options update
                self.swarm.options = self.oh(
                    self.options, iternow=i, itermax=iters
                )
                # Perform velocity and position updates
                self.swarm.velocity = self.top.compute_velocity(
                    self.swarm, self.velocity_clamp, self.vh, self.bounds
                )
                self.swarm.position = self.top.compute_position(
                    self.swarm, self.bounds, self.bh
                )
            # Obtain the final best_cost and the final best_position
            final_best_cost = self.swarm.best_cost.copy()
            final_best_pos = self.swarm.pbest_pos[
                self.swarm.pbest_cost.argmin()
            ].copy()
           
            # Close Pool of Processes
            #if n_processes is not None:
            #    pool.close()
        return (final_best_cost, final_best_pos)
    
    
def compute_objective_function(swarm, objective_func, pool, **kwargs):    
    #if pool is None:
    #    return objective_func(swarm.position, **kwargs)
    #else:
        #print('-'*50)
        #print(f'swarm.position len = {len(swarm.position)}')
        #print(f'swarm.position[4] len = {len(swarm.position[4])}')
        
        new_obj_fun = partial(objective_func, **kwargs)
        results = pool(delayed(new_obj_fun)(i) for i in swarm.position)
        
        #results = pool.map(
        #    partial(objective_func, **kwargs),
        #    np.array_split(swarm.position, pool._processes), chunksize=5
        #)
        #print(results)
        #print(f'results len = {len(results)}')
        #print('-'*50)
        return results