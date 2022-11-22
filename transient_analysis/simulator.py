import numpy as np
from pandas import Series
from scipy.stats import t, norm


class Client:
    def __init__(
            self,
            type: str,
            arrival_time: float):
        self.type = type
        self.arrival_time = arrival_time

    def get_delay(
            self,
            time: float) -> float:
        return time - self.arrival_time


class QueueSimulator:

    def hyperexponential2(
            self,
            p: np.float128,
            mu1: np.float128,
            mu2: np.float128
    ) -> np.float128:
        exp1 = lambda: self.generator.exponential(mu1)
        exp2 = lambda: self.generator.exponential(mu2)
        u = self.generator.uniform()
        return exp1() if u<p else exp2()
    
    def get_distribution(self, distribution: str):
        DISTRIBUTIONS = {
            'exp': lambda: self.generator.exponential(1),
            'det': lambda: 1,
            'hyp': lambda: self.hyperexponential2(
                                p=0.99,
                                mu1=1-1/np.sqrt(2),
                                mu2=(2+99*np.sqrt(2))/2
                                )
            }
        return DISTRIBUTIONS[distribution]

    def __init__(
            self,
            utilisation: float,
            service_distribution: str,
            transient_batch_size: int,
            steady_batch_size: int,
            transient_tolerance: float,
            confidence: float,
            accuracy: float,
            seed: int):
        """
        to-do
        """
        self.users = 0
        self.time = 0
        self.confidence = confidence
        self.transient = True
        self.transient_batch_size = transient_batch_size
        self.steady_batch_size = steady_batch_size
        self.transient_tolerance = transient_tolerance
        self.accuracy = accuracy
        self.utilisation = utilisation
        self.queue = list()
        self.fes = list()
        self.queue_sizes = list()
        self.delays = list()
        self.generator = np.random.default_rng(seed=seed)
        self.service_distribution_str = service_distribution
        self.inter_arrival_distribution = lambda: \
            self.generator.exponential(1/self.utilisation)
        self.service_distribution = self.get_distribution(service_distribution)
        # scheduling the first arrival
        self.schedule_arrival()

    def __str__(self) -> str:
        return '\nSimulator info:\n' + \
            f'Service distribution: {self.service_distribution_str}\n' + \
            f'Utilisation: {self.utilisation}'

    def schedule_event(
            self,
            time: float,
            name: str
            ) -> None:
        """
        Add an element to the Future Event Set.
        IN:
            - time: the moment at which the event will be executed.
            - name: 'arrival' to schedule an arrival event,
                    'departure' to schedule a departure event.
        OUT:
            - the new event is appended to the stateful list
              'fes', along with the action (the callable)
              to be executed at the given time.
        """
        if name == 'arrival':
            action = self.arrival
        elif name == 'departure':
            action = self.departure
        else:
            raise Exception(f'Action {name} not handled.')
        self.fes.append({
            'time': time,
            'action': action,
            'name': name
            })
    
    def schedule_arrival(self) -> None:
        """
        Schedule a new arrival event.
        IN:
            - time: the moment at which the event
                    will be executed.
        OUT:
            - the new arrival event has been added
              to the Future Event Set
        """
        self.schedule_event(
            time=self.time+self.inter_arrival_distribution(),
            name='arrival'
            )

    def schedule_departure(self) -> None:
        """
        Schedule a new departure event.
        IN:
            - time: the moment at which the event
                    will be executed.
        OUT:
            - the new departure event has been added
              to the Future Event Set
        """
        self.schedule_event(
            time=self.time+self.service_distribution(),
            name='departure'
            )

    def arrival(self) -> int:
        """
        - introduce a new user in the queue
        - schedule the next arrival
        - return the number of users in the queue
        """
        # schedule the next arrival
        self.schedule_arrival()
        # the client arrives in the queue
        self.users += 1
        self.queue_sizes.append(self.users)
        self.queue.append(Client('type1', self.time))
        if (self.users == 1):
            # there wasn't any client in the queue
            # then we serve it immediately
            self.schedule_departure()
        return self.users
              
    def departure(self) -> float:
        """
        - remove a user from the queue
        - if there was another user in the queue
           it schedules its departure
        - compute and return the delay of the removed user
        """
        removed: Client = self.queue.pop(0)
        delay = removed.get_delay(time=self.time)
        self.delays.append(delay)
        # remove the client from the queue only at departure time
        self.users -= 1
        if (self.users > 0):
            # if there was another client in the queue
            # then serve it
            self.schedule_departure()
        return delay

    def collect_batch(self, collect: str) -> None:
        """
        Collect a batch of different size basing on the state
        of the simulator (transient or steady).
        IN:
            - collect: specify which metric to collect,
                       'arrival' to store queue sizes,
                       'departure' to store delays.
        OUT:
            - The collected values are appended to
              a stateful list 'values'.
            - Return the mean delay of the batch.
        """
        batch_size = self.transient_batch_size \
            if self.transient else self.steady_batch_size
        i = 0
        batch_delay = 0
        while i < batch_size:
            self.fes.sort(key=lambda x: x['time'], reverse=True)
            next_event = self.fes.pop()
            # advance the simulator in time to execute the next event
            self.time = next_event['time']
            # execute the next event
            value = next_event['action']()
            if next_event['name'] == collect:
                self.values.append(value)
                batch_delay += value
                i += 1
        return batch_delay / batch_size

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
        values = np.array(self.batch_mean_delays)
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)/np.sqrt(n)
        if n < 30:
            return mean, t.interval(self.confidence, n-1, mean, std)
        else:
            return mean, norm.interval(self.confidence, mean, std)

    def update_cumulative_means(self) -> None:
        """
        Update the cumulative mean array of the collected metric.
        IN:
            - start: set it equal to the transient end moment
                     to recompute cumulative means only on steady
                     otherwise default 0 compute the cumulative 
                     means on transient+steady.
        OUT:
            - the cumulative means array is a field of this class,
              set equal to the transient end moment to have valuable
              results.
        """
        values = np.array(self.values)
        self.cumulative_means = Series(data=values)\
                                    .expanding()\
                                    .mean()\
                                    .values

    def exec(
            self,
            collect: str
            ) -> tuple[float, tuple[float, float]]:
        """
        Execute the simulation.
        IN:
            - collect: 'arrival' collect queue sizes.
                       'departure' collect delays.
        OUT:
            a tuple, the first value is the mean value
            of the collected metric, the second is the
            confidence interval.
        """
        # removing transient state
        self.values = list()
        transient_n = 0
        while self.transient == True:
            self.collect_batch(collect=collect)
            transient_n += 1
            self.update_cumulative_means()
            if len(self.cumulative_means) > 1:
                relative_diff = np.abs(self.cumulative_means[-1] \
                    - self.cumulative_means[-2]) / self.cumulative_means[-2]
                if relative_diff < self.transient_tolerance:
                    self.transient = False
                    self.transient_end = transient_n*self.transient_batch_size
        # collecting the first 10 batches
        steady_n = 0
        self.batch_mean_delays = list()
        while steady_n<10:
            batch_mean = self.collect_batch(collect=collect)
            self.batch_mean_delays.append(batch_mean)
            steady_n += 1
        mean, conf_int = self.confidence_interval()
        while np.abs(conf_int[0]-conf_int[1])/mean > self.accuracy:
            batch_mean = self.collect_batch(collect=collect)
            self.batch_mean_delays.append(batch_mean)
            mean, conf_int = self.confidence_interval()
            steady_n += 1
        self.update_cumulative_means()
        return mean, conf_int, transient_n, steady_n
