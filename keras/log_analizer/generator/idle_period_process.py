
from base_process import BaseProcess

class IdlePeriodProcess(BaseProcess):
    def run(self, current_time):
        if self.is_scheduled(current_time):
            return {"cpu": 2, "ram": 5, "disk": 1}
        return {"cpu": 0, "ram": 0, "disk": 0}

    def is_scheduled(self, current_time):
        return 0 <= current_time.hour < 6
