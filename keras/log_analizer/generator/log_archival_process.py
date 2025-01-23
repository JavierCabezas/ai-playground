
from base_process import BaseProcess

class LogArchivalProcess(BaseProcess):
    def run(self, current_time):
        if self.is_scheduled(current_time):
            return {"cpu": 10, "ram": 5, "disk": 50}
        return {"cpu": 0, "ram": 0, "disk": 0}

    def is_scheduled(self, current_time):
        return current_time.weekday() == 5 and current_time.hour == 3
