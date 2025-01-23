
# Log Generator with Processes

This project generates logs based on dynamic and configurable processes. Each process has its own logic for resource consumption and scheduling.

## Features
- **Multiple Process Types**: 12 processes with unique behaviors (e.g., backup, web server, high traffic).
- **Event-Driven Logs**: Simulates real-world events like server restarts and process failures.
- **Metadata Logging**: Detailed logs including CPU, RAM, and Disk usage, along with active processes.

## Usage

1. Ensure all required files are in place:
    - `base_process.py`
    - Individual process files in the same directory.

2. Run the generator to create logs:
    ```bash
    python generator.py
    ```

3. Logs will be saved in `server_logs.txt`.

## Process Types
1. **Backup Process**: Daily disk cleanup with minimal CPU/RAM.
2. **Web Server Load**: Frequent CPU/RAM usage.
3. **Database Query**: Periodic high RAM and moderate CPU usage.
4. **System Update**: Weekly high disk usage.
5. **Idle Period**: Low activity simulation.
6. **High Traffic**: Random high resource usage.
7. **Backup Verification**: Daily moderate disk usage.
8. **Log Archival**: Weekly disk-intensive operation.
9. **Disk Cleanup**: Monthly disk cleanup with negative usage.
10. **Load Balancer**: Frequent load redistribution.
11. **Monitoring Service**: Always running low-resource usage.
12. **Data Migration**: Weekly high disk and CPU usage.

## Dependencies
- Python 3.8+

---
