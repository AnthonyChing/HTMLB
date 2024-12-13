import subprocess

# List of scripts to execute sequentially
scripts = [
    "home_shared_features.py",
    "home_features.py",
    "home_combine.py",
    "away_shared_features.py",
    "away_features.py",
    "away_combine.py",
    "Late_Game_Preprocess1.py",
    "Late_Game_Preprocess2.py"
]

# Execute each script in order
for script in scripts:
    try:
        print(f"Executing {script}...")
        result = subprocess.run(["python3", script], check=True, capture_output=True, text=True)
        print(f"Execution of {script} completed successfully.")
        print(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script}.")
        print(f"Return Code: {e.returncode}")
        print(f"Error Output:\n{e.stderr}")
        break