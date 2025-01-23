class BaseProcess:
    def __init__(self, name, cpu_usage=0.0, ram_usage=0.0, disk_usage=0.0, process_type="hourly"):
        """
        process_type: Specifies when the process should run.
        Accepted values: "hourly", "daily", "weekly", "custom"
        """
        self.name = name
        self.cpu_usage = cpu_usage
        self.ram_usage = ram_usage
        self.disk_usage = disk_usage
        self.process_type = process_type

    def get_resource_usage(self):
        return self.cpu_usage, self.ram_usage, self.disk_usage

    def should_run(self, current_time):
        if self.process_type == "hourly":
            return current_time.minute == 0 and current_time.second == 0
        elif self.process_type == "daily":
            return current_time.hour == 0 and current_time.minute == 0 and current_time.second == 0
        elif self.process_type == "weekly":
            return (
                current_time.weekday() == 0 and
                current_time.hour == 0 and
                current_time.minute == 0 and
                current_time.second == 0
            )
        elif self.process_type == "custom":
            return False
        else:
            raise ValueError(f"Unknown process type: {self.process_type}")
