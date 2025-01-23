
from base_process import BaseProcess

class LoadBalancerProcess(BaseProcess):
    def run(self, current_time):
        if self.is_scheduled(current_time):
            return {"cpu": 50, "ram": 30, "disk": 5}
        return {"cpu": 0, "ram": 0, "disk": 0}

    def is_scheduled(self, current_time):
        return current_time.minute % 30 == 0
