import numpy as np
from typing import Callable
from client_management import Client, ClientQueue


class MultiServerSimulator:

    def hyperexponential2(
            self,
            p: np.float128,
            mu1: np.float128,
            mu2: np.float128) -> np.float128:
        exp1 = lambda: self.generator.exponential(mu1)
        exp2 = lambda: self.generator.exponential(mu2)
        u = self.generator.uniform()
        return exp1() if u<p else exp2()

    def get_service_time_distribution(self, type :str):
        DISTRIBUTIONS = {
            'exp': lambda: self.generator.exponential(1),
            'det': lambda: 1,
            'hyp': lambda: self.hyperexponential2(
                p=0.99,
                mu1=1-1/np.sqrt(2),
                mu2=(2+99*np.sqrt(2))/2
                )
            }
        return DISTRIBUTIONS[type]

    def __init__(
            self,
            n_servers: int,
            queue_size: int,
            service_time: str,
            inter_arrival_lp: float,
            inter_arrival_hp: float,
            seed: int):
        self.generator = np.random.default_rng(seed=seed)
        self.queue = ClientQueue(capacity=queue_size)
        self.servers = np.empty(shape=(n_servers,), dtype=Client)
        self.service_time = \
            self.get_service_time_distribution(service_time)
        self.inter_arrival = lambda priority: self.generator.poisson(
            inter_arrival_hp if priority == True else inter_arrival_lp
            )
        self.fes = list()
        self.time = 0
        self.schedule_arrival(priority=True)
        self.schedule_arrival(priority=False)

    def schedule_event(
            self,
            time: float,
            name: float,
            action: Callable):
        self.fes.append({
            'time': time,
            'action': action,
            'name': name
            })
        
    def schedule_arrival(self, priority: bool):
        self.schedule_event(
            time = self.time + self.inter_arrival(priority=priority),
            name = 'arrival_hp' if priority == True else 'arrival_lp',
            action = self.arrival_hp if priority == True else self.arrival_lp
            )

    def schedule_departure(self):
        self.schedule_event(
            time = self.time + self.service_time(),
            name = 'departure',
            action = self.departure
        )

    def arrival_lp(self):
        self.schedule_arrival(priority=False)
        

    def arrival_hp(self):
        self.schedule_arrival(priority=True)

    def departure(self):
        pass
