
from base_process import BaseProcess

class MonitoringService(BaseProcess):
    def run(self, current_time):
        return {"cpu": 5, "ram": 5, "disk": 2}  # Always running
