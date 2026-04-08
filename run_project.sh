#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# Master Control Script: Full Project Initialization & Execution
# ═══════════════════════════════════════════════════════════════════════════
# This script handles complete project setup and execution:
#   1. Environment configuration
#   2. Dependency installation
#   3. ML pipeline training
#   4. Dashboard initialization
#   5. Dashboard launch
# ═══════════════════════════════════════════════════════════════════════════

set -e

# ─────────────────────────────────────────────────────────────────────────
# Colors for output
# ─────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────
clear
echo -e "${PURPLE}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║                                                                            ║${NC}"
echo -e "${PURPLE}║     🏥 RARE DISEASE ML - REAL-TIME CLINICAL DASHBOARD                     ║${NC}"
echo -e "${PURPLE}║                    Master Control Script                                    ║${NC}"
echo -e "${PURPLE}║                                                                            ║${NC}"
echo -e "${PURPLE}╚════════════════════════════════════════════════════════════════════════════╝${NC}"

# ─────────────────────────────────────────────────────────────────────────
# Get script directory
# ─────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "\n${CYAN}📍 Working Directory:${NC} $SCRIPT_DIR\n"

# ─────────────────────────────────────────────────────────────────────────
# Step 1: Check Python Installation
# ─────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/6]${NC} Checking Python installation…"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo -e "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"

# ─────────────────────────────────────────────────────────────────────────
# Step 2: Set up Virtual Environment
# ─────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[2/6]${NC} Setting up virtual environment…"

if [ ! -d ".venv" ]; then
    echo -e "  Creating new virtual environment…"
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# ─────────────────────────────────────────────────────────────────────────
# Step 3: Install Dependencies
# ─────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[3/6]${NC} Installing dependencies…"

# Upgrade pip
python -m pip install --upgrade pip > /dev/null 2>&1

# Install requirements
echo -e "  Installing packages from requirements.txt…"
pip install -q -r requirements.txt 2>&1 | tee /tmp/pip_install.log || {
    # Check if dowhy was the issue (Python 3.14 incompatibility)
    if grep -q "dowhy" /tmp/pip_install.log; then
        echo -e "${YELLOW}⚠ DoWhy skipped (Python 3.14 incompatibility). Using fallback causal analysis.${NC}"
        # Install everything except dowhy
        pip install -q numpy pandas scipy scikit-learn xgboost \
            imbalanced-learn shap dice-ml matplotlib seaborn \
            streamlit tqdm joblib
    else
        echo -e "${RED}✗ Dependency installation failed${NC}"
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────
# Step 4: Create Project Directories
# ─────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[4/6]${NC} Setting up project directories…"

mkdir -p data
mkdir -p outputs
mkdir -p outputs/archive

echo -e "${GREEN}✓ Directories created${NC}"

# ─────────────────────────────────────────────────────────────────────────
# Step 5: Check and Train Models (if needed)
# ─────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[5/6]${NC} Checking ML models…"

if [ ! -f "outputs/trained_models.pkl" ]; then
    echo -e "  ${CYAN}ℹ Models not found. Training ML pipeline…${NC}"
    echo -e "  This may take 2-5 minutes on first run…\n"
    
    # Check if Kaggle data exists
    if [ ! -f "data/kaggle_data.csv" ]; then
        echo -e "${YELLOW}⚠ Warning:${NC} Kaggle dataset not found at ${CYAN}data/kaggle_data.csv${NC}"
        echo -e "  ${CYAN}ℹ Using demo data generation mode${NC}"
    fi
    
    # Run the main pipeline
    if python main.py; then
        echo -e "\n${GREEN}✓ ML pipeline completed successfully${NC}"
    else
        echo -e "${RED}✗ ML pipeline failed${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Trained models found${NC}"
    
    # Check for all required artifacts
    all_artifacts=true
    for artifact in "trained_preprocessor.pkl" "feature_names.pkl" "best_threshold.pkl"; do
        if [ ! -f "outputs/$artifact" ]; then
            all_artifacts=false
            break
        fi
    done
    
    if [ "$all_artifacts" = false ]; then
        echo -e "  ${YELLOW}⚠ Some dashboard artifacts missing. Regenerating…${NC}"
        python main.py > /dev/null 2>&1
        echo -e "${GREEN}✓ Artifacts regenerated${NC}"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────
# Step 6: Validate Dashboard App
# ─────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[6/6]${NC} Validating dashboard app…"

if [ -f "app.py" ]; then
    echo -e "${GREEN}✓ Dashboard app found (app.py)${NC}"
else
    echo -e "${RED}✗ app.py not found${NC}"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────
# Launch Dashboard
# ─────────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Setup Complete! Starting Dashboard…${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"

echo -e "\n${CYAN}📊 Dashboard Information:${NC}"
echo -e "  ${CYAN}URL:${NC} http://localhost:8501"
echo -e "  ${CYAN}Port:${NC} 8501"
echo -e "  ${CYAN}Press Ctrl+C to stop${NC}"
echo -e "\n${YELLOW}📖 Quick Start:${NC}"
echo -e "  1. Open http://localhost:8501 in your browser"
echo -e "  2. Use the sidebar to set baseline patient inputs"
echo -e "  3. Use Simulation Lab to test input changes"
echo -e "  4. Check ⚠️ Alerts for any critical patients"
echo -e ""

# Launch Streamlit dashboard
streamlit run app.py --logger.level=warning
