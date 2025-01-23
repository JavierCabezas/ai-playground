
from base_process import BaseProcess

class BackupProcess(BaseProcess):
    def run(self, current_time):
        if self.is_scheduled(current_time):
            return {"cpu": 5, "ram": 10, "disk": -20}
        return {"cpu": 0, "ram": 0, "disk": 0}

    def is_scheduled(self, current_time):
        return current_time.hour == 0 and current_time.minute == 0
