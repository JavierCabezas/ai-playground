
from base_process import BaseProcess

class DataMigrationProcess(BaseProcess):
    def run(self, current_time):
        if self.is_scheduled(current_time):
            return {"cpu": 40, "ram": 30, "disk": 60}
        return {"cpu": 0, "ram": 0, "disk": 0}

    def is_scheduled(self, current_time):
        return current_time.weekday() == 0 and current_time.hour == 22
