# events.py

import random

# Event type-specific probabilities and impacts
EVENTS = {
    "Unexpected Spike": {
        "probability": 0.0002,  # 0.02%
        "cpu_range": (80, 100),
        "ram_range": (90, 100),
        "disk_increase": 10,
    },
    "Server Restart": {
        "probability": 0.0001,  # 0.01%
        "cpu_range": (0, 5),  # Low CPU after restart
        "ram_range": (0, 10),
        "disk_increase": 0,
    },
    "Process Failure": {
        "probability": 0.00015,  # 0.015%
        "cpu_range": (50, 70),
        "ram_range": (50, 70),
        "disk_increase": 5,
    },
}

def should_trigger_event(event_type):
    """Determines if an event of a specific type should be triggered."""
    return random.random() < EVENTS[event_type]["probability"]

def apply_event(event_type, cpu, ram, disk):
    """Applies the impact of an event to the given metrics."""
    event = EVENTS[event_type]
    cpu = round(random.uniform(*event["cpu_range"]), 1)
    ram = round(random.uniform(*event["ram_range"]), 1)
    disk = min(round(disk + event["disk_increase"], 1), 100)  # Cap disk at 100%
    return cpu, ram, disk
