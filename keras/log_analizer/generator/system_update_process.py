
from base_process import BaseProcess

class SystemUpdateProcess(BaseProcess):
    def run(self, current_time):
        if self.is_scheduled(current_time):
            return {"cpu": 30, "ram": 10, "disk": 40}
        return {"cpu": 0, "ram": 0, "disk": 0}

    def is_scheduled(self, current_time):
        return current_time.weekday() == 6 and current_time.hour == 2
