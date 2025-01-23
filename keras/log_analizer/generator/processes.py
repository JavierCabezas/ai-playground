from base_process import BaseProcess
from custom_process import CustomProcess

PROCESSES = [
    BaseProcess("HourlyProcess", cpu_usage=5.0, ram_usage=2.0, process_type="hourly"),
    BaseProcess("DailyBackup", cpu_usage=10.0, ram_usage=5.0, disk_usage=20.0, process_type="daily"),
    BaseProcess("WeeklyReport", cpu_usage=3.0, ram_usage=1.0, process_type="weekly"),
    CustomProcess("CustomTask", cpu_usage=7.0, ram_usage=3.0),
]
