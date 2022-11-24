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


class ClientQueue:
    def __init__(self, capacity: int):
        self.high_priority_queue = list()
        self.low_priority_queue = list()
        self.capacity = capacity
        self.size = 0
        self.low_priority_size = 0
        self.high_priority_size = 0

    def append_high_priority(self, arrival_time: int) -> None:
        if self.size == self.capacity:
            # if there is no space then
            # remove the last arrived low priority client
            self.low_priority_queue.pop()
            self.size -= 1
            self.low_priority_size -= 1
        self.high_priority_queue.append(Client(
            type='high_priority',
            arrival_time=arrival_time
            ))
        self.size += 1
        self.high_priority_size += 1

    def append_low_priority(self, arrival_time: int) -> bool:
        if self.size == self.capacity: return False
        self.low_priority_queue.append(Client(
            type='low_priority',
            arrival_time=arrival_time
            ))
        self.size += 1
        self.low_priority_size += 1
        return True

    def append(self,
            priority: bool,
            arrival_time: int) -> bool:
        if priority == True:
            self.append_high_priority(arrival_time=arrival_time)
            return True
        else:
            return self.append_low_priority(arrival_time=arrival_time)
    
    def pop(self, priority: bool) -> Client:
        if priority == True:
            client = self.high_priority_queue.pop(0)
            self.high_priority_size -= 1
        else:
            client = self.low_priority_queue.pop(0)
            self.low_priority_size -= 1
        self.size -= 1
        return client
