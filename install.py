import subprocess
import sys
import os

requirements = os.path.join(os.path.dirname(__file__), "env.yaml")
conda_executable = "/root/miniconda3/bin/conda"

try:
    subprocess.check_call([conda_executable, "env", "update", "--file", requirements])
    print("Successfully installed.")
except subprocess.CalledProcessError as e:
    print(f"Failed installing: {e}")
    sys.exit(1)