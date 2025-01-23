from base_process import BaseProcess

class CustomProcess(BaseProcess):
    def __init__(self, name, cpu_usage=0.0, ram_usage=0.0, disk_usage=0.0):
        super().__init__(name, cpu_usage, ram_usage, disk_usage, process_type="custom")

    def should_run(self, current_time):
        # Custom logic: Run every 15 minutes
        return current_time.minute % 15 == 0 and current_time.second == 0
