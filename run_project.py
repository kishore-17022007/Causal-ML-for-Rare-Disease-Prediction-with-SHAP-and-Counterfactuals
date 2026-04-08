#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
from pathlib import Path

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

def log(level, msg):
    if level == "error":
        print(f"{Colors.RED}✗ {msg}{Colors.NC}")
    elif level == "success":
        print(f"{Colors.GREEN}✓ {msg}{Colors.NC}")
    elif level == "warning":
        print(f"{Colors.YELLOW}⚠ {msg}{Colors.NC}")
    elif level == "info":
        print(f"{Colors.CYAN}ℹ {msg}{Colors.NC}")
    elif level == "step":
        print(f"{Colors.YELLOW}{msg}{Colors.NC}")
    else:
        print(msg)

def print_banner():
    """Print welcome banner"""
    print(f"\n{Colors.PURPLE}╔════════════════════════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.PURPLE}║                                                                            ║{Colors.NC}")
    print(f"{Colors.PURPLE}║     🏥 RARE DISEASE ML - REAL-TIME CLINICAL DASHBOARD                     ║{Colors.NC}")
    print(f"{Colors.PURPLE}║                    Master Control Script (Python)                          ║{Colors.NC}")
    print(f"{Colors.PURPLE}║                                                                            ║{Colors.NC}")
    print(f"{Colors.PURPLE}╚════════════════════════════════════════════════════════════════════════════╝{Colors.NC}\n")

def check_python():
    """Check Python version"""
    log("step", "[1/6] Checking Python installation…")
    
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 8):
        log("error", f"Python 3.8+ required, found {sys.version}")
        sys.exit(1)
    
    log("success", f"Python {version_info.major}.{version_info.minor}.{version_info.micro} found")

def setup_venv():
    """Set up virtual environment"""
    log("step", "\n[2/6] Setting up virtual environment…")
    
    venv_path = Path(".venv")
    
    if not venv_path.exists():
        log("info", "Creating new virtual environment…")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        log("success", "Virtual environment created")
    else:
        log("success", "Virtual environment exists")

def install_dependencies():
    """Install project dependencies"""
    log("step", "\n[3/6] Installing dependencies…")
    
    # Upgrade pip
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )
    
    # Install requirements (with graceful fallback for incompatible packages)
    log("info", "Installing packages from requirements.txt…")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # If dowhy fails (Python 3.14 incompatibility), install everything else
    if result.returncode != 0 and "dowhy" in result.stderr:
        log("warning", "DoWhy skipped (Python 3.14 incompatibility). Using fallback causal analysis.")
        # Install all core packages except dowhy
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "numpy>=1.24", "pandas>=2.0", "scipy>=1.10",
            "scikit-learn>=1.3", "xgboost>=1.7",
            "imbalanced-learn>=0.11", "shap>=0.42",
            "dice-ml>=0.9", "matplotlib>=3.7", "seaborn>=0.12",
            "streamlit>=1.28", "tqdm>=4.65", "joblib>=1.3"
        ], check=True)
    elif result.returncode != 0:
        log("error", f"Dependency installation failed: {result.stderr[:200]}")
        sys.exit(1)
    
    log("success", "All dependencies installed")

def create_directories():
    """Create project directories"""
    log("step", "\n[4/6] Setting up project directories…")
    
    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/archive").mkdir(exist_ok=True)
    
    log("success", "Directories created")

def check_kaggle_data():
    """Check if Kaggle data exists"""
    if not Path("data/kaggle_data.csv").exists():
        log("warning", "Kaggle dataset not found at data/kaggle_data.csv")
        log("info", "Using demo data generation mode")
        return False
    return True

def train_models():
    """Train ML models"""
    log("step", "\n[5/6] Checking ML models…")
    
    models_exist = Path("outputs/trained_models.pkl").exists()
    
    if not models_exist:
        log("info", "Models not found. Training ML pipeline…")
        log("info", "This may take 2-5 minutes on first run…\n")
        
        check_kaggle_data()
        
        try:
            subprocess.run([sys.executable, "main.py"], check=True)
            log("success", "ML pipeline completed successfully")
        except subprocess.CalledProcessError:
            log("error", "ML pipeline failed")
            sys.exit(1)
    else:
        log("success", "Trained models found")
        
        # Check for all required artifacts
        artifacts = [
            "trained_preprocessor.pkl",
            "feature_names.pkl",
            "best_threshold.pkl"
        ]
        
        missing = [a for a in artifacts if not Path(f"outputs/{a}").exists()]
        
        if missing:
            log("warning", "Some dashboard artifacts missing. Regenerating…")
            subprocess.run(
                [sys.executable, "main.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            log("success", "Artifacts regenerated")

def initialize_dashboard():
    """Validate dashboard app entrypoint exists."""
    log("step", "\n[6/6] Validating dashboard app…")
    if Path("app.py").exists():
        log("success", "Dashboard app found (app.py)")
    else:
        log("error", "Dashboard app file app.py not found")
        sys.exit(1)

def launch_dashboard():
    """Launch Streamlit dashboard"""
    log("success", "═══════════════════════════════════════════════════════════════════════════")
    log("success", "✓ Setup Complete! Starting Dashboard…")
    log("success", "═══════════════════════════════════════════════════════════════════════════")
    
    print(f"\n{Colors.CYAN}📊 Dashboard Information:{Colors.NC}")
    print(f"  {Colors.CYAN}URL:{Colors.NC} http://localhost:8501")
    print(f"  {Colors.CYAN}Port:{Colors.NC} 8501")
    print(f"  {Colors.CYAN}Press Ctrl+C to stop{Colors.NC}")
    
    print(f"\n{Colors.YELLOW}📖 Quick Start:{Colors.NC}")
    print(f"  1. Open http://localhost:8501 in your browser")
    print(f"  2. Use the sidebar to set baseline patient inputs")
    print(f"  3. Use Simulation Lab to test input changes")
    print(f"  4. Check ⚠️ Alerts for any critical patients\n")
    
    # Launch Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", 
                   "--logger.level=warning"])

def main():
    """Main entry point"""
    print_banner()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"{Colors.CYAN}📍 Working Directory:{Colors.NC} {script_dir}\n")
    
    try:
        # Execute all steps
        check_python()
        setup_venv()
        install_dependencies()
        create_directories()
        train_models()
        initialize_dashboard()
        launch_dashboard()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user${Colors.NC}")
        sys.exit(0)
    except Exception as e:
        log("error", f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
