import os
import subprocess
import sys
import venv

# 1. We create a virtual enviroment called "Medicion con DAQ" (Measurement with DAQ in spanish), change at will
env_name = "Medicion Con DAQ" 
env_dir = os.path.abspath(env_name)

print(f"🔧 Creating virtual environment: {env_dir}")
venv.create(env_dir, with_pip=True) # We use pip
print("✅ Virtual environment created.")

# 2. Define list of required packages
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
]

# 3. Install packages
python_bin = os.path.join(env_dir, "bin", "python") if os.name != "nt" else os.path.join(env_dir, "Scripts", "python.exe")

print("Installing required packages...")
for pkg in required_packages:
    subprocess.check_call([python_bin, "-m", "pip", "install", pkg])

print("All packages installed.")
print("If it all worked, it should have created a new enviroment at {env_dir}")
print("Make sure to use this enviroment when running the acquisition code")
#print(f"\nTo activate the environment:\n\nsource '{env_dir}/bin/activate'  # Linux/macOS\n")
