import psutil

def print_system_metrics():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    print(f"CPU Usage: {cpu_usage:.2f}%, Memory Usage: {memory_usage:.2f}%")
