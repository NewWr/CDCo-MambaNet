from typing import List
from sklearn.metrics import accuracy_score
import numpy as np

class WeightedMeter:
    def __init__(self, name: str = None):
        self.name = name
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0
        self.val = 0.0
        self.max_value = float('-inf')  

    def update(self, val: float, num: int = 1):
        self.count += num
        self.sum += val * num
        self.avg = self.sum / self.count
        self.val = val
        if self.val > self.max_value:
            self.max_value = self.val 

    def reset(self, total: float = 0, count: int = 0):
        self.count = count
        self.sum = total
        self.avg = total / max(count, 1)
        self.val = total / max(count, 1)
        


class AverageMeter:
    def __init__(self, length: int, name: str = None):
        assert length > 0
        self.name = name
        self.count = 0
        self.sum = 0.0
        self.current: int = -1
        self.history: List[float] = [None] * length

    @property
    def val(self) -> float:
        return self.history[self.current]

    @property
    def avg(self) -> float:
        return self.sum / self.count

    def update(self, val: float):
        self.current = (self.current + 1) % len(self.history)
        self.sum += val

        old = self.history[self.current]
        if old is None:
            self.count += 1
        else:
            self.sum -= old
        self.history[self.current] = val


class TotalMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.max_acc = 0

    def update(self, val: float):
        self.sum += val
        self.count += 1

    def update_with_weight_loss(self, val: float, count: int):
        self.sum += val*count
        self.count += count

    def update_with_weight_acc(self, val: list, label: list):
        predicted_labels = np.argmax(val, axis=1)
        label = np.argmax(label, axis=1)
        accuracy = accuracy_score(label, predicted_labels)
        if self.max_acc < accuracy:
            self.max_acc = accuracy

    def reset(self):
        self.sum = 0
        self.count = 0
        self.max_acc = 0

    @property
    def avg(self):
        if self.count == 0:
            return -1
        return self.sum / self.count

    @property
    def final(self):
        return self.max_acc
    
