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
        self.fes.append(
            self.FutureEvent(
                time=time,
                action=action,
                name=name,
                client=client
            ))
        
    def schedule_arrival(self, priority: bool) -> None:
        client = Client(
            id=self.__get_next_id__(),
            priority=priority,
            arrival_time=self.time +\
                self.inter_arrival_distribution(priority=priority),
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

    def arrival(
            self,
            priority: bool,
            client: Client) -> tuple[float, float]:
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
        return self.queue.high_priority_size, self.queue.low_priority_size

    def departure(self, client_id: int) -> float|None:
        if client_id not in self.to_skip_departures:
            client = self.servers.find_client(client_id)
            if client is not None:
                client, position = client
                self.served_clients += int(not self.transient)
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
            
    def confidence_interval(
            self,
            data: np.ndarray
            ) -> tuple[float, float, float]:
        """
        Compute the confidence interval of the mean value
        of the collected metric, from the start value
        IN:
            - None
        OUT:
            - the mean value
            - the left confidence interval
            - the right confidence interval
        """
        data = data[self.transient_end:]
        n = len(data)
        mean = float(np.mean(data))
        std = np.std(data, ddof=1)/np.sqrt(n)
        interval = t.interval(self.confidence, n-1, mean, std) if n < 30 \
            else norm.interval(self.confidence, mean, std)
        return mean, interval[0], interval[1]

    @staticmethod
    def cumulative_mean(data: np.ndarray) -> np.ndarray:
        return np.array(
            Series(data=data).expanding().mean().values
            )

    def collect_batch(self) -> dict:
        """
        Collect a batch of a size choosed looking at the
        state of the simulator (transient/steady).
        """
        queue_size = list()
        queue_size_hp = list()
        queue_size_lp = list()
        delay = list()
        delay_hp = list()
        delay_lp = list()
        batch_size = self.transient_batch_size if self.transient \
            else self.steady_batch_size
        while len(queue_size) < batch_size \
                or len(delay) < batch_size:
            self.fes.sort(key=lambda x: x.time, reverse=True)
            next_event: MultiServerSimulator.FutureEvent = \
                self.fes.pop()
            self.time = next_event.time
            if next_event.name.startswith('arrival'):
                hp_value, lp_value = next_event.action()
                if len(queue_size) < batch_size:
                    queue_size.append(hp_value+lp_value)
                if len(queue_size_hp) < batch_size:
                    queue_size_hp.append(hp_value)
                if len(queue_size_lp) < batch_size:
                    queue_size_lp.append(lp_value)
            elif next_event.name == 'departure':
                value: float|None = next_event.action()
                if value is not None:
                    delay.append(value)
                    (delay_hp if next_event.client.priority \
                        else delay_lp).append(value)

        return {
            'queue_size': np.array(queue_size),
            'queue_size_hp': np.array(queue_size_hp),
            'queue_size_lp': np.array(queue_size_lp),
            'delay': np.array(delay),
            'delay_hp': np.array(delay_hp),
            'delay_lp': np.array(delay_lp)
        }

    def execute(self) -> dict:
        """
        It executes the simulation using the parameters provided to
        the constructor of this class.
        """
        self.served_clients = 0
        values = dict()
        cumulative_means = dict()
        n_batches = 0
        means = {
            'mean_delay': [],
            'mean_delay_hp': [],
            'mean_delay_lp': [],
            'mean_queue_size': [],
            'mean_queue_size_hp': [],
            'mean_queue_size_lp': []
        }
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
                cumulative_means[key] = self.cumulative_mean(
                    data=values[key]
                    )
            cum_mean = cumulative_means['queue_size_hp']
            rel_diff = np.abs(cum_mean[-1]-cum_mean[-2])/cum_mean[-2]
            if rel_diff < self.transient_tolerance:
                self.transient = False
                self.transient_end = n_batches
            if not self.transient:
                means['mean_delay'].append(
                    self.confidence_interval(batch['delay'])
                    )
                means['mean_delay_hp'].append(
                    self.confidence_interval(batch['delay_hp'])
                    )
                means['mean_delay_lp'].append(
                    self.confidence_interval(batch['delay_lp'])
                    )
                means['mean_queue_size'].append(
                    self.confidence_interval(batch['queue_size'])
                    )
                means['mean_queue_size_hp'].append(
                    self.confidence_interval(batch['queue_size_hp'])
                    )
                means['mean_queue_size_lp'].append(
                    self.confidence_interval(batch['queue_size_lp'])
                    )
        means_np = dict()
        accuracy = dict()
        for key in means:
            means_np[key] = np.array(means[key])
            last = means_np[key][-1]
            accuracy[key] = np.abs(last[1]-last[2])/last[0]

        return {
            'n_batches': n_batches,
            'values': values,
            'cumulative_mean': cumulative_means,
            'means': means_np,
            'accuracy': accuracy
        }
