import os
import subprocess
import sys
import venv

# 1. Name of virtual environment
env_name = "Medicion Con DAQ"
env_dir = os.path.abspath(env_name)

print(f"🔧 Creating virtual environment: {env_dir}")
venv.create(env_dir, with_pip=True)
print("✅ Virtual environment created.")

# 2. Required packages based on your imports
required_packages = [
    "numpy==2.3.0",
    "pandas==2.2.3",
    "matplotlib==3.10.3",
    "PyQt5==5.15.11",
    "PyQtWebEngine==5.15.7",
    "pyqtgraph==0.13.4",
    "nidaqmx==0.6.2",
    "scipy==1.13.1",
    "numba==0.59.1",
    "psutil==7.0.0",
    "scikit-learn",
    "soundfile",
    "pygame",
    "datetime",  # built-in, no need to install
    "csv",       # built-in, no need to install
    "glob2",     # glob is built-in, but glob2 is installable and useful
]

python_bin = os.path.join(env_dir, "bin", "python") if os.name != "nt" else os.path.join(env_dir, "Scripts", "python.exe")

print("📦 Installing required packages...")
for pkg in required_packages:
    subprocess.check_call([python_bin, "-m", "pip", "install", pkg])

print("✅ All packages installed.")
print(f"📁 Environment created at: {env_dir}")
print("🔌 Make sure to activate this environment when running the acquisition code.")
