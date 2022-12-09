import numpy as np


class GWSimulator:
    def __init__(
            self,
            lam: float,
            max_gen_nodes: int,
            seed: int):
        self.max_gen_nodes = max_gen_nodes
        self.lam = lam
        self.generator = np.random.default_rng(seed=seed)
        self.extract_n_child = lambda: self.generator.poisson(lam=lam)
        self.n_generations = 0
        self.generation_nodes = 1
        self.total_nodes = 1

    def execute(self) -> int:
        while self.generation_nodes > 0 \
        and self.generation_nodes < self.max_gen_nodes:
            n_child_prev = self.generation_nodes
            self.generation_nodes = 0
            for _ in range(n_child_prev):
                self.generation_nodes += self.extract_n_child()
            self.n_generations += 1
            self.total_nodes += self.generation_nodes
        return self.n_generations if self.generation_nodes == 0 else -1
