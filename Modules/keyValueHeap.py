import heapq


class keyValueHeap():

    def __init__(self, size):
        self.heap = []
        heapq.heapify(self.heap)
        self.size = size


    def length(self):
        return len(self.heap)

    def push(self, key, value):
        if self.length() >= self.size:
            if self.peek()[0] < key:
                self.pop()
            else:
                return

        heapq.heappush(self.heap, (key,value))

    def peek(self):
        key, value = heapq.heappop(self.heap)
        heapq.heappush(self.heap, (key, value))
        return key, value

    def pop(self):
        key, value = heapq.heappop(self.heap)
        return key, value

    def merge(self, items):
        for key, value in items.items():
            self.push(key, value)

    def print(self):
      temp = []
      for _ in range(self.length()):
        key, value = self.pop()
        temp.append((key, value))
        print(key, value)
      
      for t in temp:
        heapq.heappush(self.heap, t)

    def reverse(heap):
        temp = []
        for _ in range(heap.length()):
            key, value = heap.pop()
            temp.insert(0, (value, abs(key)))
        return temp

