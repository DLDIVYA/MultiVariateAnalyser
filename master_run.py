import subprocess
import sys


required_libraries = [
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "seaborn"
]

def install_missing_packages():
    """
    Install any missing Python packages automatically.
    """
    for package in required_libraries:
        try:
            __import__(package)
        except ImportError:
            print(f"[INFO] Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def run_script(command, step_name):
    """
    Run a Python script with error handling.
    """
    print(f"\n[INFO] Running {step_name}...")
    try:
        subprocess.run(command, check=True)
        print(f"[SUCCESS] {step_name} completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {step_name} failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Step 0: Ensure required packages are installed
    install_missing_packages()

    # Step 1: Run MultivariateAnalyserCheck.py
    run_script(["python", "MultivariateAnalyserCheck.py"], "MultivariateAnalyserCheck.py")

    # Step 2: Run generate_plots.py
    run_script(["python", "generate_plots.py"], "generate_plots.py")

    # Step 3: Run alerts.py with arguments
    run_script([
        "python", "alerts.py",
        "--input_csv", "TEP_Train_Test_with_anomalies.csv",
        "--output_csv", "TEP Test_with_alerts.csv",
        "--timestamp_col", "Time"
    ], "alerts.py")

    print("All scripts executed successfully!")
