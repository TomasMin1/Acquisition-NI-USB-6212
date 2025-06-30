import os
import subprocess
import sys

# 1. Define environment name
env_name = "medicion-con-daq"

# 2. Conda packages (available from conda-forge or defaults)
conda_packages = [
    "python=3.12",
    "numpy=2.3.0",
    "pandas=2.2.3",
    "matplotlib=3.10.3",
    "pyqt=5.15.11",
    "pyqtgraph=0.13.4",
    "scipy=1.13.1",
    "numba=0.59.1",
    "psutil=7.0.0",
    "scikit-learn",
    "soundfile",
    "pygame"
]

# 3. pip-only packages
pip_packages = [
    "nidaqmx==0.6.2"
]

# 4. Construct the full command
create_env_cmd = [
    "conda", "create", "-y", "-n", env_name
] + conda_packages

print("🔧 Creating conda environment...")
subprocess.check_call(create_env_cmd)
print("✅ Conda environment created.")

# 5. Install pip-only packages inside the conda environment
print("📦 Installing pip-only packages...")

# Build the pip install command
pip_install_cmd = f"""
conda run -n {env_name} python -m pip install {' '.join(pip_packages)}
"""

subprocess.check_call(pip_install_cmd, shell=True)
print("✅ All pip packages installed.")

# 6. Final instructions
print(f"\n🎉 Environment '{env_name}' is ready!")
print(f"👉 To activate it, run:\n\nconda activate {env_name}\n")
