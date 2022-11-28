import numpy as np
from pandas import Series
from typing import Callable
from client_server_queue import Client, ClientPriorityQueue
from scipy.stats import t, norm
import math


class MultiServerSimulator:
    
    class FutureEvent:
        def __init__(
                self,
                time: float,
                action: Callable,
                name: str,
                client: Client):
            self.time = time
            self.action = action
            self.name = name
            self.client = client

    def hyperexponential2(
            self,
            p: float,
            mu1: float,
            mu2: float) -> float:
        exp1 = lambda: self.generator.exponential(mu1)
        exp2 = lambda: self.generator.exponential(mu2)
        u = self.generator.uniform()
        return exp1() if u<p else exp2()

    def service_time_distribution(
            self,
            priority: bool) -> float:
        DISTRIBUTIONS_A = {
            'exp': lambda: self.generator.exponential(1),
            'det': lambda: 1,
            'hyp': lambda: self.hyperexponential2(
                p=0.99,
                mu1=1-1/math.sqrt(2),
                mu2=(2+99*math.sqrt(2))/2
                )
            }
        DISTRIBUTION_B_HP = {
            'exp': lambda: self.generator.exponential(0.5),
            'det': lambda: 0.5,
            'hyp': lambda: self.hyperexponential2(
                p=0.99,
                mu1 = (2-math.sqrt(2))/4,
                mu2 = (2+99*math.sqrt(2))/math.sqrt(2)
            )
        }
        DISTRIBUTION_B_LP = {
            'exp': lambda: self.generator.exponential(1.5),
            'det': lambda: 1.5,
            'hyp': lambda: self.hyperexponential2(
                p=0.99,
                mu1 = (6-3*math.sqrt(2))/4,
                mu2 = (6+97*math.sqrt(2))/4
            )
        }
        distribution = DISTRIBUTIONS_A if self.service_time_case == 'a' \
            else DISTRIBUTION_B_HP if priority else DISTRIBUTION_B_LP
        return distribution[self.service_time_distribution_str]()

    def inter_arrival_distribution(self, priority: bool) -> float:
        exp_hp = lambda: self.generator.exponential(
            1/self.inter_arrival_hp_lambda
            )
        exp_lp = lambda: self.generator.exponential(
            1/self.inter_arrival_lp_lambda
            )
        return exp_hp() if priority else exp_lp()

    def __init__(
            self,
            n_servers: int,
            queue_size: int,
            service_time_distribution: str,
            inter_arrival_lp_lambda: float,
            inter_arrival_hp_lambda: float,
            service_time_case: str,
            steady_batch_size: int,
            transient_batch_size: int,
            transient_tolerance: float,
            confidence: float,
            max_served_clients: int,
            seed: int):
        self.n_servers = n_servers
        self.queue_size = queue_size
        self.service_time_distribution_str = service_time_distribution
        self.inter_arrival_hp_lambda = inter_arrival_hp_lambda
        self.inter_arrival_lp_lambda = inter_arrival_lp_lambda
        self.steady_batch_size = steady_batch_size
        self.transient_batch_size = transient_batch_size
        self.transient_tolerance = transient_tolerance
        self.max_served_clients = max_served_clients
        self.confidence = confidence
        self.service_time_case = service_time_case
        self.transient = True
        self.time = 0
        self.next_id = 0
        self.generator = np.random.default_rng(seed=seed)
        self.queue = ClientPriorityQueue(capacity=queue_size)
        self.servers = ClientPriorityQueue(capacity=n_servers)
        self.fes = list()
        self.to_skip_departures = dict()
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
        new_event = \
            self.FutureEvent(
                time=time,
                action=action,
                name=name,
                client=client
            )
        self.fes.append(new_event)
        
    def schedule_arrival(self, priority: bool) -> None:
        client = Client(
            id=self.__get_next_id__(),
            priority=priority,
            arrival_time=self.time + self.inter_arrival_distribution(priority=priority),
            service_time=self.service_time_distribution(priority=priority),
            start_service_time=-1
            )
         
        self.schedule_event(
            time = client.arrival_time,
            name = 'arrival_hp' if priority == True else 'arrival_lp',
            action = lambda: self.arrival(
                priority=priority,
                client=client
                ),
            client = client
            )

    def schedule_departure(self, client: Client) -> None:
        self.schedule_event(
            time = self.time + client.service_time,
            name = 'departure',
            action = lambda: self.departure(client.id),
            client = client
        )

    def arrival(self, priority: bool, client: Client) -> float:
        self.schedule_arrival(priority=priority)
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
                        if removed_low_priority.id in self.to_skip_departures.keys():
                            self.to_skip_departures[removed_low_priority.id] += 1
                        else:
                            self.to_skip_departures[removed_low_priority.id] = 1
        return self.queue.size

    def departure(self, client_id: int) -> float|None:
        if client_id not in self.to_skip_departures:
            client = self.servers.find_client(client_id)
            if client is not None:
                client, position = client
                self.served_clients += 1#int(not self.transient)
                self.servers.pop_specific_client(
                    priority=client.priority,
                    position=position
                    )
                if not self.queue.is_empty():
                    next_client = self.queue.pop()
                    self.servers.append(next_client)
                    self.schedule_departure(next_client)
                return client.get_delay(self.time)
            else:
                raise Exception('Performing departure on None')
        else:
            self.to_skip_departures[client_id] -= 1
            if self.to_skip_departures[client_id] == 0:
                self.to_skip_departures.pop(client_id)
            
    @staticmethod
    def confidence_interval(
            data: np.ndarray,
            confidence: float
            ) -> tuple[float, tuple[float, float]]:
        """
        Compute the confidence interval of the mean value
        of the collected metric, from the 'start' value.
        IN:
            - None
        OUT:
            - the mean value
            - the confidence interval
        """
        n = len(data)
        mean = float(np.mean(data))
        std = np.std(data, ddof=1)/np.sqrt(n)
        if n < 30:
            return mean, t.interval(confidence, n-1, mean, std)
        else:
            return mean, norm.interval(confidence, mean, std)

    @staticmethod
    def cumulative_mean(data: np.ndarray) -> np.ndarray:
        return np.array(
            Series(data=data).expanding().mean().values
            )

    def collect_batch(self) -> dict:
        n_sample_queue_size = 0
        n_sample_delay = 0
        batch_size = self.transient_batch_size if self.transient \
            else self.steady_batch_size
        values_queue_size = np.empty(shape=(batch_size,), dtype=float)
        values_hp_queue_size = list()
        values_lp_queue_size = list()
        values_delay = np.empty(shape=(batch_size,), dtype=float)
        values_hp_delay = list()
        values_lp_delay = list()
        while n_sample_queue_size < batch_size \
        or n_sample_delay < batch_size:
            self.fes.sort(
                key=lambda event: event.time,
                reverse=True
                )
            next_event: MultiServerSimulator.FutureEvent = \
                self.fes.pop()
            self.time = next_event.time
            value = next_event.action()
            if next_event.name.startswith('arrival') \
            and n_sample_queue_size < batch_size:
                values_queue_size[n_sample_queue_size] = value
                (values_hp_queue_size if next_event.client.priority \
                     else values_lp_queue_size).append(value)
                n_sample_queue_size += 1
            if next_event.name == 'departure' \
            and value is not None\
            and n_sample_delay < batch_size:
                values_delay[n_sample_delay] = value
                (values_hp_delay if next_event.client.priority \
                     else values_lp_delay).append(value)
                n_sample_delay += 1

        return {
            'queue_size': values_queue_size,
            'queue_size_hp': np.array(values_hp_queue_size),
            'queue_size_lp': np.array(values_lp_queue_size),
            'delay': values_delay,
            'delay_hp': np.array(values_hp_delay),
            'delay_lp': np.array(values_lp_delay)
        }

    def execute(self) -> tuple[dict, dict, int]:
        self.served_clients = 0
        values = dict()
        means = dict()
        n_batches = 0
        while self.served_clients < self.max_served_clients:
            batch: dict = self.collect_batch()
            n_batches += 1
            if len(values) > 0:
                for key in batch:
                    values[key] = np.concatenate((values[key], batch[key]))
            else:
                values = batch
            for key in values:
                means[key] = self.cumulative_mean(data=values[key])
        return values, means, n_batches
