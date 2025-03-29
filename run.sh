#!/bin/bash

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
  echo "requirements.txt not found!"
  exit 1
fi

# Check for the existence of a virtual environment folder (usually named 'venv')
if [ -d "venv" ]; then
  echo "Virtual environment already exists."
else
  echo "Creating virtual environment..."
  
  # Create virtual environment based on OS
  if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    python3 -m venv venv
  elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    python -m venv venv
  else
    echo "Unsupported OS type: $OSTYPE"
    exit 1
  fi
  echo "Virtual environment created."
fi

# Activate the virtual environment and install dependencies
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
  source venv/bin/activate
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
  source venv/Scripts/activate
fi

# Install the requirements from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! Virtual environment is active and dependencies are installed."

# Check if something is already running on port 3000 and kill the process (Windows)
if [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
  # Get the PID by parsing netstat output more reliably for Windows
  PID=$(netstat -ano | findstr :3000 | awk '{print $5}' | sed 's/[^0-9]*//g' | head -n 1)  # Extract PID cleanly
  if [ -n "$PID" ]; then
    echo "Something is already running on port 3000. Stopping $PID..."
    powershell.exe -Command "Stop-Process -Id $PID -Force"
    echo "Process on port 3000 stopped."
  else
    echo "No process running on port 3000."
  fi
fi

# Build and start the docker image
echo "Fetch and build docker image"
docker rm -f ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull llama3.2:1b

# Run the backend app.py in the background
echo "Running backend/app.py..."
python backend/app.py &

# Navigate to the frontend directory and start React
echo "Running React frontend..."
cd frontend || { echo "Frontend directory not found!"; exit 1; }
# Remember to install Node.js before this step!
npm install  # Install frontend dependencies if not already done
npm start &

# Wait for both processes to finish
wait
