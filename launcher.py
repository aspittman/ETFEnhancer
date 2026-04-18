import subprocess
import sys
import time
from datetime import datetime

while True:
    print("\nStarting trading bot...")
    result = subprocess.run([sys.executable, "main.py"])

    if result.returncode == 0:
        print("Bot exited normally. Not restarting.")
        break

    with open("crash.log", "a") as f:
        f.write(f"{datetime.now()} - Bot crashed with return code {result.returncode}\n")

    print("Bot crashed. Restarting in 30 seconds...")
    time.sleep(30)