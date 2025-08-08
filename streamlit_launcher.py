# streamlit_launcher.py
# Simple launcher script to run the Streamlit app with proper configuration

import os
import sys
import subprocess

def run_streamlit_app():
    """
    Launch the Streamlit app with proper configuration to minimize warnings
    """
    # Set environment variables to reduce warnings
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(current_dir, "PowerBIDynamicScatterPlotVisualizer.py")

    # Check if the main app file exists
    if not os.path.exists(app_file):
        print(f"Error: Could not find PowerBIDynamicScatterPlotVisualizer.py in {current_dir}")
        print("Please make sure both files are in the same directory.")
        sys.exit(1)
    
    # Launch Streamlit with configuration
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_file,
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    
    print("🚀 Starting BI Dashboard...")
    print(f"📁 App file: {app_file}")
    print("🌐 Opening browser at: http://localhost:8501")
    print("\n" + "="*50)
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down BI Dashboard...")
        print("Thank you for using the application!")

if __name__ == "__main__":
    run_streamlit_app()