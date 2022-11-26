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

    def __str__(self) -> str:
        return str(self.id)

    def get_delay(
            self,
            time: float) -> float:
        return time - self.arrival_time


class ClientPriorityQueue:

    class PriorityQueueException(Exception):
        pass

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

    def __str__(self) -> str:
        low = ''
        high = ''
        for i in range(self.low_priority_size):
            c = self.low_priority_queue[i]
            low = low + str(c) + ' '
        for i in range(self.high_priority_size):
            c = self.high_priority_queue[i]
            high = high + str(c) + ' '
        return 'low: ' + low + '\nhigh: ' + high

    def is_available(self) -> bool:
        return self.size < self.capacity

    def pop(self, front: bool = True) -> Client:
        """
        Return the oldest client with highest priority, if exists
             otherwise raise an exception.
        IN: front: true pop from the front, false pop from the back
        OUT: the oldest client with highest priority.
        """
        pop_high = self.pop_front_high_priority if front \
            else self.pop_back_high_priority
        pop_low = self.pop_front_low_priority if front \
            else self.pop_back_low_priority
        if self.size == 0:
            raise self.PriorityQueueException('No client in the queue')
        try:
            client = pop_high()
        except self.PriorityQueueException:
            client = pop_low()
        self.size -= 1
        return client

    def append(
            self,
            client: Client,
            force: bool = False,
            front: bool = False) -> tuple[bool, Client|None]:
        """
        Append a new client at the beginning of the appropriate queue.
        If the client is high priority and the queue is full, try to
        remove the youngest low priority client.
        If it is not possible and you don't force, the push fails.
        IN:
            - client: a new client to insert at the end of the queue.
            - front: true append in the front, false append in the back.
            - force: if True and the queue is full, remove
            the last arrived client in the queue with
            the same priority.
        OUT:
            - return True iff the client was inserted. Furthermore, if a
            low priority client has been removed to make space to a high
            priority one, it is returned.
        """
        increment = int(self.size < self.capacity)
        removed_low_priority = None
        push_low_priority = self.push_front_low_priority if front \
            else self.push_back_low_priority
        push_high_priority = self.push_front_high_priority if front \
            else self.push_back_high_priority
        if client.priority:
            if self.size == self.capacity:
                try:
                    removed_low_priority = self.pop_back_low_priority()
                    push_high_priority(client, force)
                    inserted = True
                except self.PriorityQueueException:
                    try:
                        push_high_priority(client, force)
                        inserted = True
                    except self.PriorityQueueException:
                        inserted = False
            else:
                self.push_front_high_priority(client, force)
                inserted = True
        else:
            if self.high_priority_size == self.capacity:
                inserted = False
            else:
                try:
                    push_low_priority(client, force)
                    inserted = True
                except self.PriorityQueueException:
                    inserted = False
        self.size += increment
        return inserted, removed_low_priority

    def roll_high_priority(self, shift: int) -> None:
        self.high_priority_queue = np.roll(
            a=self.high_priority_queue,
            shift=shift
        )
    
    def roll_low_priority(self, shift: int) -> None:
        self.low_priority_queue = np.roll(
            a=self.low_priority_queue,
            shift=shift
        )

    def pop_front_high_priority(self) -> Client:
        """
        Return the oldest client with high priority if exists,
        otherwise raise an exception.
        IN: None
        OUT: the oldest client with high priority
        """
        if self.high_priority_size == 0:
            raise self.PriorityQueueException('No high priority client in the queue.')
        client = self.high_priority_queue[0]
        self.roll_high_priority(-1)
        self.high_priority_size -= 1
        return client

    def pop_front_low_priority(self) -> Client:
        """
        Return the oldest client with low priority if exists,
        otherwise raise an exception.
        IN: None
        OUT: the oldest client with low priority
        """
        if self.low_priority_size == 0:
            raise self.PriorityQueueException('No low priority client in the queue.')
        client = self.low_priority_queue[0]
        self.roll_low_priority(-1)
        self.low_priority_size -= 1
        return client

    def pop_back_high_priority(self) -> Client:
        """
        Return the youngest client with high priority, if exists
        otherwise raise an exception
        IN: None
        OUT: the youngest client with high priority
        """
        if self.high_priority_size == 0:
            raise self.PriorityQueueException('No high priority client in the queue')
        self.high_priority_size -= 1
        return self.high_priority_queue[self.high_priority_size]

    def pop_back_low_priority(self) -> Client:
        """
        Return the youngest client with low priority, if exists
        otherwise raise an exception
        IN: None
        OUT: the youngest client with low priority
        """
        if self.low_priority_size == 0:
            raise self.PriorityQueueException('No low priority client in the queue')
        self.low_priority_size -= 1
        return self.low_priority_queue[self.low_priority_size]

    def pop_back(self) -> Client:
        """
        Return the youngest client with lowest prioriy, if exists
            otherwise raise an exception
        IN: None
        OUT: the youngest client with lowest priority
        """
        if self.size == 0:
            raise self.PriorityQueueException('No client in the queue')
        try:
            client = self.pop_back_low_priority()
        except self.PriorityQueueException:
            client = self.pop_back_high_priority()
        self.size -= 1
        return client

    def push_back_low_priority(
            self,
            client: Client,
            force: bool) -> None:
        """
        Append a client at the end of the low priority queue.
        If it is full and you don't force, an exception is raised.
        IN:
            - client: a new client to insert at the end of the
                      low priority queue
            - force: if True and the queue is full, remove
            the last arrived client from the low priority
            queue (if exists)
        OUT: None
        """
        if client.priority:
            raise Exception('Illegal operation')
        if self.low_priority_size == self.capacity:
            if force:
                self.low_priority_queue[self.low_priority_size-1] = client
            else:
                raise self.PriorityQueueException('Full low priority queue')
        else:
            self.low_priority_queue[self.low_priority_size] = client
            self.low_priority_size += 1

    def push_back_high_priority(
            self,
            client: Client,
            force: bool) -> None:
        """
        Append a client to the end of the high priority queue.
        If it is full, try to remove a low priority client.
        If it is not possible and you don't force, an
        exception is raised.
        IN:
            - client: a new client to insert at the end of the
                      high priority queue
            - force: if True and the queue is full, remove
            the last arrived client from the high priority
            queue (if exists)
        OUT: None
        """
        if not client.priority:
            raise Exception('Illegal operaton')
        if self.high_priority_size == self.capacity:
            try:
                self.pop_back_low_priority()
            except self.PriorityQueueException:
                if force:
                    self.high_priority_queue[self.high_priority_size-1] = client
                else:
                    raise self.PriorityQueueException('Full high priority queue')
        else:
            self.high_priority_queue[self.high_priority_size] = client
            self.high_priority_size += 1

    def push_front_low_priority(
            self,
            client: Client,
            force: bool) -> None:
        """
        Append a new client at the beginning of the low priority queue.
        If the queue is full and you don't force, an exception is raised.
        IN:
            - client: a new client to insert at the end of the queue.
            - force: if True and the queue is full, remove
            the last arrived client in the low priority queue
        OUT: None
        """
        if client.priority:
            raise Exception('Illegal operaton')
        if self.low_priority_size == self.capacity:
            if force:
                self.roll_low_priority(1)
                self.low_priority_queue[0] = client
            else:
                raise self.PriorityQueueException('Full low priority queue')
        else:
            self.roll_low_priority(1)
            self.low_priority_queue[0] = client
            self.low_priority_size += 1

    def push_front_high_priority(
            self,
            client: Client,
            force: bool) -> None:
        """
        Append a new client at the beginning of the high priority queue.
        If the queue is full and you don't force, raise an exception.
        IN:
            - client: a new client to insert at the end of the queue.
            - force: if True and the queue is full, remove
            the last arrived client in the high priority queue
        OUT:
            - return True iff the client was inserted.
        """
        if not client.priority:
            raise Exception('Illegal operaton')
        if self.high_priority_size == self.capacity:
            if force:
                self.roll_high_priority(1)
                self.high_priority_queue[0] = client
            else:
                raise self.PriorityQueueException('Full high priority queue')
        else:
            self.roll_high_priority(1)
            self.high_priority_queue[0] = client
            self.high_priority_size += 1