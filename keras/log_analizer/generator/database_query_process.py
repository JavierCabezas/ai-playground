
from base_process import BaseProcess

class DatabaseQueryProcess(BaseProcess):
    def run(self, current_time):
        if self.is_scheduled(current_time):
            return {"cpu": 20, "ram": 50, "disk": 10}
        return {"cpu": 0, "ram": 0, "disk": 0}

    def is_scheduled(self, current_time):
        return current_time.hour % 4 == 0 and current_time.minute == 0
