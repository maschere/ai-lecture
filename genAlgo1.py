#%% defs
import random
from typing import Callable
random_state = 124
random.seed(random_state)

# function to minimize
def f(x, y):
    if x ** 2 + y ** 2 > 2:
        return 1e10
    else:
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


# formulation as fitness to maximize
def fitness(x, y):
    return 1 + -1 * f(x, y)


class indi:
    """individium class"""

    def __init__(self, func: Callable):
        """init individum with random x,y in [-2,2]

        Args:
            func (Callable): fitness funciton taking x,y params

        """
        self.x = (random.random() - 0.5) * 4
        self.y = (random.random() - 0.5) * 4
        self.fitness = func

    def eval(self) -> float:
        """evaluate fitness of this individuum

        Returns:
            [float]: fitness score (higher better)
        """
        return self.fitness(self.x, self.y)

    def mutate(self, sigma=0.01):
        """mutate by drawing from ndist around current value with sigma"""
        self.x = random.normalvariate(self.x, sigma)
        self.y = random.normalvariate(self.y, sigma)


def xover(a: indi, b: indi) -> indi:
    """crossover between two individuals by randomly-weighted linear interpolation between their respective coefficients

    Args:
        a (indi): parent a
        b (indi): parent b

    Returns:
        indi: child c
    """
    c = indi(fitness)
    rel = random.random()
    c.x = rel * a.x + (1 - rel) * b.x
    c.y = rel * b.y + (1 - rel) * a.y
    return c


#%% run it
popsize = 100
maxgen = 500
use_elitism = True
allow_self_reproduction = True
pop = [indi(fitness) for i in range(popsize)]

for gen in range(maxgen):
    pop.sort(key=lambda p0: p0.eval(), reverse=True)
    best = pop[0]
    print(f"{gen}: fitness: {best.eval()} with f({best.x},{best.y})={f(best.x,best.y)}")

    # cross over top 10 indis of old pop
    pop = pop[0:10]
    new_pop = []
    for a in pop:
        for b in pop:
            if allow_self_reproduction == False and a == b:
                continue
            new_ind = xover(a, b)
            new_ind.mutate()
            new_pop.append(new_ind)
    if use_elitism:
        pop = pop[0:1] + new_pop
    else:
        pop = new_pop

# %% logging