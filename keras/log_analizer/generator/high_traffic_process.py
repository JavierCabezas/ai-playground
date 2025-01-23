
from base_process import BaseProcess
import random

class HighTrafficProcess(BaseProcess):
    def run(self, current_time):
        if self.is_scheduled(current_time):
            return {"cpu": 70, "ram": 50, "disk": 10}
        return {"cpu": 0, "ram": 0, "disk": 0}

    def is_scheduled(self, current_time):
        return random.random() < 0.05  # 5% chance each time step
