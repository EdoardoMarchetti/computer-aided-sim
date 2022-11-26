import numpy as np


class Client:
    def __init__(
            self,
            id: int,
            priority: bool,
            arrival_time: float,
            service_time: float,
            start_service_time: float):
        self.id = id
        self.priority = priority
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.start_service_time = start_service_time

    def get_delay(
            self,
            time: float) -> float:
        return time - self.arrival_time


class ClientPriorityQueue:
    def __init__(self, capacity: int):
        """
        Create a new queue with two levels of priority
        (low and high). Capacity is the parameter to
        specify the maximum number of clients in the
        queue
        """
        self.high_priority_queue = np.empty(
            shape=(capacity,),
            dtype=Client
            )
        self.low_priority_queue = np.empty(
            shape=(capacity,),
            dtype=Client
            )
        self.capacity = capacity
        self.size = 0
        self.low_priority_size = 0
        self.high_priority_size = 0

    def push_high_priority(
            self,
            client: Client,
            front: bool) -> bool:
        """
        Push a new Client in the high priority queue.
        IN:
            - client: the object to be inserted.
            - front: True to push in the front,
                     False to push in the back.
        OUT: True iff the client was successfully pushed.
        """
        if self.high_priority_size == self.capacity:
            return False
        if self.size == self.capacity:
            self.pop_back(priority=False)
        if not front:
            self.high_priority_queue = np.roll(
                a=self.high_priority_queue,
                shift=1
                )
            self.high_priority_queue[0] = client
        else:
            self.high_priority_queue[self.high_priority_size] = client
        self.high_priority_size += 1
        self.size += 1
        return True

    def push_low_priority(
            self,
            client: Client,
            front: bool,
            force: bool = False) -> bool:
        """
        Push a new client in the low priority queue.
        IN:
            - client: the object to be inserted.
            - front: True to push in the front,
                     False to push in the back.
        OUT: True iff the client was successfully pushed.
        """
        increment = self.size < self.capacity
        if increment or force:
            if front:
                self.low_priority_queue = np.roll(
                    a=self.low_priority_queue,
                    shift=1
                )
                self.low_priority_queue[0] = client
            else:
                self.low_priority_queue[self.low_priority_size] = client
            self.low_priority_size += increment
            self.size += increment
            return True
        return False

    def append(
            self,
            client: Client) -> bool:
        """
        Append a new client at the end of the queue.
        Returns True iff the client was successfully pushed.
        """
        action = self.push_high_priority \
            if client.priority is True else self.push_low_priority
        return action(client=client, front=False)
    
    def pop(self) -> Client:
        """
        Removes and returns the first customer with highest priority.
        """
        priority = self.high_priority_size > 0
        return self.pop_front(priority=priority)
        

    def pop_back(self, priority: bool) -> Client:
        """
        Remove the last customer with the specified priority.
        IN:
            - priority: True for high priority, False for low.
        OUT: the removed client object.
        """
        if priority == True:
            self.high_priority_size -= 1
            client = self.high_priority_queue[self.high_priority_size]
        else:
            self.low_priority_size -= 1
            client = self.low_priority_queue[self.low_priority_queue]
        self.size -= 1
        return client

    def pop_front(self, priority: bool) -> Client:
        """
        Remove the first customer with the specified priority.
        IN:
            - priority: True for high priority, False for low.
        OUT: the removed client object.
        """
        if priority == True:
            client = self.high_priority_queue[0]
            self.high_priority_queue = np.roll(
                a=self.high_priority_queue,
                shift=-1
                )
            self.high_priority_size -= 1
        else:
            client = self.low_priority_queue[0]
            self.low_priority_queue = np.roll(
                a=self.low_priority_queue,
                shift=-1
                )
            self.low_priority_size -= 1
        self.size -= 1
        return client


class MultiServer:

    class OnlyHighPriorityException(Exception):
        pass

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.server = np.empty(shape=(capacity,), dtype=Client)
        self.size = 0

    def is_available(self) -> bool:
        return self.size < self.capacity

    def submit(self, client: Client) -> bool:
        if self.is_available():
            self.server[self.size] = client
            self.size += 1
            return True
        return False

    def remove_latest_lp(self) -> Client:
        max_i = 0
        max_arrival = -1
        for i in range(self.size):
            client: Client = self.server[i]
            if client.priority == False \
            and client.arrival_time > max_arrival:
                max_arrival = client.arrival_time
                max_i = i
        if max_arrival < 0:
            raise self.OnlyHighPriorityException()
        max_client: Client = self.server[max_i]
        self.size -= 1
        self.server[max_i] = self.server[self.size]
        return max_client
