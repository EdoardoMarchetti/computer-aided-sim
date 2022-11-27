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
                mu1 = 0.5 - 7*math.sqrt(11)/66,
                mu2 = 0.5 + 21*math.sqrt(11)/2
            )
        }
        DISTRIBUTION_B_LP = {
            'exp': lambda: self.generator.exponential(1.5),
            'det': lambda: 1.5,
            'hyp': lambda: self.hyperexponential2(
                p=0.99,
                mu1 = 1.5 - math.sqrt(4917)/66,
                mu2 = 1.5 + 3*math.sqrt(4917)/2
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
            accuracy: float,
            seed: int):
        self.n_servers = n_servers
        self.queue_size = queue_size
        self.service_time_distribution_str = service_time_distribution
        self.inter_arrival_hp_lambda = inter_arrival_hp_lambda
        self.inter_arrival_lp_lambda = inter_arrival_lp_lambda
        self.steady_batch_size = steady_batch_size
        self.transient_batch_size = transient_batch_size
        self.transient_tolerance = transient_tolerance
        self.accuracy = accuracy
        self.confidence = confidence
        self.service_time_case = service_time_case
        self.transient = True
        self.time = 0
        self.next_id = 0
        self.generator = np.random.default_rng(seed=seed)
        self.queue = ClientPriorityQueue(capacity=queue_size)
        self.servers = ClientPriorityQueue(capacity=n_servers)
        self.fes = list()
        self.to_skip_departures = set()
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
                        self.to_skip_departures.add(removed_low_priority.id)
        return self.queue.size

    def departure(self, client_id: int) -> float|None:
        if client_id not in self.to_skip_departures:
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
                return client.get_delay(self.time)
            else:
                raise Exception('Performing departure on None')
        else:
            self.to_skip_departures.remove(client_id)
            
    def collect_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_sample = 0
        batch_size = self.transient_batch_size if self.transient \
            else self.steady_batch_size
        values = np.empty(shape=(batch_size,), dtype=float)
        values_hp, values_lp = list(), list()
        while n_sample < batch_size:
            self.fes.sort(
                key=lambda event: event.time,
                reverse=True
                )
            next_event: MultiServerSimulator.FutureEvent = \
                self.fes.pop()
            self.time = next_event.time
            value = next_event.action()
            if next_event.name == 'departure' \
            and value is not None:
                values[n_sample] = value
                (values_hp if next_event.client.priority \
                     else values_lp).append(value)
                n_sample += 1
        return values, np.array(values_lp), np.array(values_hp)

    def confidence_interval(self) -> tuple[float, tuple[float, float]]:
        """
        Compute the confidence interval of the mean value
        of the collected metric, from the 'start' value.
        IN:
            - None
        OUT:
            - the mean value
            - the confidence interval
        """
        values = np.array(self.steady_means)
        n = len(values)
        mean = float(np.mean(values))
        std = np.std(values, ddof=1)/np.sqrt(n)
        if n < 30:
            return mean, t.interval(self.confidence, n-1, mean, std)
        else:
            return mean, norm.interval(self.confidence, mean, std)

    @staticmethod
    def cumulative_mean(data: np.ndarray) -> np.ndarray:
        return np.array(
            Series(data=data).expanding().mean().values
            )

    def execute(self):
        transient_batches = 0
        self.transient_values = np.empty(shape=0)
        self.transient_values_lp = np.empty(shape=0)
        self.transient_values_hp = np.empty(shape=0)
        self.transient_means = list()
        self.transient_means_hp = list()
        self.transient_means_lp = list()
        while self.transient == True:
            batch, batch_lp, batch_hp = self.collect_batch()
            self.transient_values = np.concatenate((self.transient_values, batch))
            self.transient_values_lp = np.concatenate((self.transient_values_lp, batch_lp))
            self.transient_values_hp = np.concatenate((self.transient_values_hp, batch_hp))
            self.transient_means.append(np.mean(batch))
            self.transient_means_hp.append(np.mean(batch_hp))
            self.transient_means_lp.append(np.mean(batch_lp))
            transient_batches += 1
            self.cumulative_means = self.cumulative_mean(data=self.transient_values)
            self.cumulative_means_hp = self.cumulative_mean(data=self.transient_values_hp)
            self.cumulative_means_lp = self.cumulative_mean(data=self.transient_values_lp)
            relative_diff = np.abs(
                self.cumulative_means[-1] - self.cumulative_means[-2]) \
                / self.cumulative_means[-2]
            if relative_diff < self.transient_tolerance:
                self.transient = False
                self.transient_end = transient_batches*self.transient_batch_size
        steady_batches = 0
        self.steady_values_lp = np.empty(shape=0)
        self.steady_values_hp = np.empty(shape=0)
        self.steady_values = np.empty(shape=0)
        self.steady_means = list()
        self.steady_means_lp = list()
        self.steady_means_hp = list()
        while steady_batches < 10:
            batch, batch_lp, batch_hp = self.collect_batch()
            self.steady_means.append(np.mean(batch))
            self.steady_means_lp.append(np.mean(batch_lp))
            self.steady_means_hp.append(np.mean(batch_hp))
            self.steady_values = np.concatenate((self.steady_values, batch))
            self.steady_values_lp = np.concatenate((self.steady_values_lp, batch_lp))
            self.steady_values_hp = np.concatenate((self.steady_values_hp, batch_hp))
            steady_batches += 1
        mean, conf_int = self.confidence_interval()
        while np.abs(conf_int[0]-conf_int[1]) / mean > self.accuracy:
            batch, batch_lp, batch_hp = self.collect_batch()
            self.steady_means.append(np.mean(batch))
            self.steady_means_lp.append(np.mean(batch_lp))
            self.steady_means_hp.append(np.mean(batch_hp))
            self.steady_values = np.concatenate((self.steady_values, batch))
            self.steady_values_lp = np.concatenate((self.steady_values_lp, batch_lp))
            self.steady_values_hp = np.concatenate((self.steady_values_hp, batch_hp))
            steady_batches += 1
            mean, conf_int = self.confidence_interval()
        self.cumulative_means = self.cumulative_mean(
            data=np.concatenate((self.transient_values, self.steady_values))
            )
        self.cumulative_means_hp = self.cumulative_mean(
            data=np.concatenate((self.transient_values_hp, self.steady_values_hp))
        )
        self.cumulative_means_lp = self.cumulative_mean(
            data=np.concatenate((self.transient_values_lp, self.steady_values_lp))
        )
                