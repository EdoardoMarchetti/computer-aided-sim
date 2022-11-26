import numpy as np
from typing import Callable
from custom_structures import \
    Client, ClientPriorityQueue, MultiServer
import math


class MultiServerSimulator:
    
    def hyperexponential2(
            self,
            p: float,
            mu1: float,
            mu2: float) -> float:
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
                mu1=1-1/math.sqrt(2),
                mu2=(2+99*math.sqrt(2))/2
                )
            }
        return DISTRIBUTIONS[type]

    def __init__(
            self,
            n_servers: int,
            queue_size: int,
            service_time: str,
            inter_arrival_lp_lambda: float,
            inter_arrival_hp_lambda: float,
            seed: int):
        self.generator = np.random.default_rng(seed=seed)
        self.queue = ClientPriorityQueue(capacity=queue_size)
        self.servers = MultiServer(capacity=n_servers)
        self.service_time = \
            self.get_service_time_distribution(service_time)
        self.inter_arrival = lambda priority: \
            self.generator.exponential(1/inter_arrival_hp_lambda) \
            if priority == True else self.generator.exponential(1/inter_arrival_lp_lambda)
        self.fes = list()
        # to_skip_departures' size will be at maximum
        # equal to the number of servers
        self.to_skip_departures = set()
        self.n_servers = n_servers
        self.time = 0
        self.next_id = 0
        self.schedule_arrival(priority=True)
        self.schedule_arrival(priority=False)

    def __get_next_id__(self) -> int:
        self.next_id += 1
        return self.next_id

    def schedule_event(
            self,
            time: float,
            name: str,
            action: Callable,
            client_id: int):
        self.fes.append({
            'time': time,
            'action': action,
            'name': name,
            'client_id': client_id
            })
        
    def schedule_arrival(self, priority: bool):
        client_id = self.__get_next_id__()
        self.schedule_event(
            time = self.time + self.inter_arrival(priority=priority),
            name = 'arrival_hp' if priority == True else 'arrival_lp',
            action = lambda: self.arrival(priority=priority, client_id=client_id),
            client_id = client_id
            )

    def schedule_departure(self, client: Client):
        self.schedule_event(
            time = self.time + client.service_time,
            name = 'departure',
            action = self.departure,
            client_id = client.id
        )

    def arrival(self, priority: bool, client_id: int):
        self.schedule_arrival(priority=priority)
        service_time = self.service_time()
        client = Client(
            id=client_id,
            priority=priority,
            arrival_time=self.time,
            service_time=service_time,
            start_service_time=-1
        )
        self.queue.append(client)
        if self.servers.is_available():
            client.start_service_time = self.time
            self.servers.submit(client)
            self.schedule_departure(client=client)
        elif priority is True:
        # liberare un server
            # selezionare il cliente in esecuzione con low priority e con tempo arrivo massimo
            # rimuoverlo dal server
            # inserire nel server l'high priority
            # aggiornare service time cliente rimosso
            # inserirlo in fronte alla coda di clienti
            # inserire id cliente nel set di departure da skippare
            try:
                lp_client = self.servers.remove_latest_lp()
            except MultiServer.OnlyHighPriorityException:
                return
            client.start_service_time = self.time
            self.servers.submit(client)
            residual_service_time = self.time - lp_client.start_service_time
            lp_client.service_time = residual_service_time
            self.queue.push_low_priority(
                client=lp_client,
                front=True,
                force=True
                )
            self.to_skip_departures.add(lp_client.id)

    def departure(self):
        pass
    
    def execute(self):
        pass
