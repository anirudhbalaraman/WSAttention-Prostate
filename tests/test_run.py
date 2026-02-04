import subprocess
import sys
from pathlib import Path


def test_run_pirads_training():
    """
    Test that run_cspca.py runs without crashing using an existing YAML config.
    """

    # Path to your run_pirads.py script
    repo_root = Path(__file__).parent.parent
    script_path = repo_root / "run_pirads.py"

    # Path to your existing config.yaml
    config_path = repo_root / "config" / "config_pirads_train.yaml"  # adjust this path

    # Make sure the file exists
    assert config_path.exists(), f"Config file not found: {config_path}"

    # Run the script with the config
    result = subprocess.run(
        [sys.executable, str(script_path), "--mode", "train", "--config", str(config_path), "--dry_run", "True" ],
        capture_output=True,
        text=True,
    )

    # Check that it ran without errors
    assert result.returncode == 0, f"Script failed with:\n{result.stderr}"

def test_run_pirads_inference():
    """
    Test that run_cspca.py runs without crashing using an existing YAML config.
    """

    # Path to your run_pirads.py script
    repo_root = Path(__file__).parent.parent
    script_path = repo_root / "run_pirads.py"

    # Path to your existing config.yaml
    config_path = repo_root / "config" / "config_pirads_test.yaml"  # adjust this path

    # Make sure the file exists
    assert config_path.exists(), f"Config file not found: {config_path}"

    # Run the script with the config
    result = subprocess.run(
        [sys.executable, str(script_path), "--mode", "test", "--config", str(config_path), "--dry_run", "True" ],
        capture_output=True,
        text=True,
    )

    # Check that it ran without errors
    assert result.returncode == 0, f"Script failed with:\n{result.stderr}"
    
def test_run_cspca_training():
    """
    Test that run_cspca.py runs without crashing using an existing YAML config.
    """

    # Path to your run_cspca.py script
    repo_root = Path(__file__).parent.parent
    script_path = repo_root / "run_cspca.py"

    # Path to your existing config.yaml
    config_path = repo_root / "config" / "config_cspca_train.yaml"  # adjust this path

    # Make sure the file exists
    assert config_path.exists(), f"Config file not found: {config_path}"

    # Run the script with the config
    result = subprocess.run(
        [sys.executable, str(script_path), "--mode", "train", "--config", str(config_path), "--dry_run", "True" ],
        capture_output=True,
        text=True,
    )

    # Check that it ran without errors
    assert result.returncode == 0, f"Script failed with:\n{result.stderr}"
    
def test_run_cspca_inference():
    """
    Test that run_cspca.py runs without crashing using an existing YAML config.
    """

    # Path to your run_cspca.py script
    repo_root = Path(__file__).parent.parent
    script_path = repo_root / "run_cspca.py"

    # Path to your existing config.yaml
    config_path = repo_root / "config" / "config_cspca_test.yaml"  # adjust this path

    # Make sure the file exists
    assert config_path.exists(), f"Config file not found: {config_path}"

    # Run the script with the config
    result = subprocess.run(
        [sys.executable, str(script_path), "--mode", "test", "--config", str(config_path), "--dry_run", "True" ],
        capture_output=True,
        text=True,
    )

    # Check that it ran without errors
    assert result.returncode == 0, f"Script failed with:\n{result.stderr}"


