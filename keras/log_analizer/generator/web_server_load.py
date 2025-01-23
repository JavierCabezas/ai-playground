
from base_process import BaseProcess
import random

class WebServerLoad(BaseProcess):
    def run(self, current_time):
        if self.is_scheduled(current_time):
            return {"cpu": random.uniform(30, 50), "ram": random.uniform(20, 40), "disk": 5}
        return {"cpu": 0, "ram": 0, "disk": 0}

    def is_scheduled(self, current_time):
        return current_time.minute % 15 == 0
