import numpy as np
from typing import Callable
from custom_structures import Client, ClientPriorityQueue
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
            endtime: int,
            seed: int):
        self.generator = np.random.default_rng(seed=seed)
        self.queue = ClientPriorityQueue(capacity=queue_size)
        self.servers = ClientPriorityQueue(capacity=n_servers)
        self.endtime = endtime
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
            client: Client) -> None:
        self.fes.append({
            'time': time,
            'action': action,
            'name': name,
            'client': client
            })
        
    def schedule_arrival(self, priority: bool) -> None:
        client = Client(
            id=self.__get_next_id__(),
            priority=priority,
            arrival_time=self.time + self.inter_arrival(priority=priority),
            service_time=self.service_time(),
            start_service_time=-1
            )
         
        self.schedule_event(
            time = client.arrival_time,
            name = 'arrival_hp' if priority == True else 'arrival_lp',
            action = lambda: self.arrival(priority=priority, client_id=client.id),
            client = client
            )

    def schedule_departure(self, client: Client) -> None:
        self.schedule_event(
            time = self.time + client.service_time,
            name = 'departure',
            action = lambda: self.departure(client.id),
            client = client
        )

    def arrival(self, priority: bool, client_id: int) -> None:
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
            client = self.queue.pop()
            client.start_service_time = self.time
            submitted, removed_low_priority = self.servers.append(client)
            if submitted:
                self.schedule_departure(client)
                if removed_low_priority is not None:
                    removed_low_priority.service_time = \
                        self.time - removed_low_priority.start_service_time
                    rescheduled, _ = self.queue.append(
                        client=removed_low_priority,
                        front=True,
                        force=True
                        )
                    if rescheduled:
                        self.to_skip_departures.add(removed_low_priority.id)

    def departure(self, client_id: int) -> None:
        if client_id in self.to_skip_departures:
            self.to_skip_departures.remove(client_id)
            return
        client = self.servers.find_client(client_id)
        if client is not None:
            client, position = client
            self.servers.pop_specific_client(
                position=position,
                priority=client.priority
                )
            if not self.queue.is_empty():
                next_client = self.queue.pop()
                self.servers.append(next_client)
                self.schedule_departure(next_client)
        else:
            raise Exception('Performing departure on None')
            
    def execute(self):
        while self.time < self.endtime:
            self.fes.sort(key=lambda x: x['time'], reverse=True)
            next_event = self.fes.pop()
            self.time = next_event['time']
            next_event['action']()
