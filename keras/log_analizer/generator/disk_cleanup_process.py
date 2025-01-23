
from base_process import BaseProcess

class DiskCleanupProcess(BaseProcess):
    def run(self, current_time):
        if self.is_scheduled(current_time):
            return {"cpu": 10, "ram": 5, "disk": -30}
        return {"cpu": 0, "ram": 0, "disk": 0}

    def is_scheduled(self, current_time):
        return current_time.day == 1 and current_time.hour == 4
