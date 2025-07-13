#!/usr/bin/env python3
"""
Deployment helper script for Neural Data Analysis App
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def setup_deployment():
    print("Setting up git repository...")
    run_command("git init", "Git init")
    run_command("git add .", "Git add all files")
    run_command("git commit -m 'Initial commit'", "Git commit")
    print("Git repository initialized.")

def deploy_to_streamlit():
    print("Deploying to Streamlit Cloud...")
    print("1. Push your code to GitHub.")
    print("2. Go to https://share.streamlit.io and follow the deployment wizard.")
    print("3. Set up secrets and config as needed.")
    print("4. Enjoy your deployed app!")

def main():
    """Main deployment function"""
    print("🧠 Neural Data Analysis App - Deployment Helper")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            setup_deployment()
        elif command == "streamlit":
            deploy_to_streamlit()
        elif command == "heroku":
            print("🚀 Heroku deployment:")
            print("1. Install Heroku CLI")
            print("2. Run: heroku create your-app-name")
            print("3. Run: git push heroku main")
            print("4. Run: heroku open")
        else:
            print(f"❌ Unknown command: {command}")
    else:
        print("Usage:")
        print("  python deploy.py setup     - Setup git repository")
        print("  python deploy.py streamlit - Deploy to Streamlit Cloud")
        print("  python deploy.py heroku    - Deploy to Heroku")

if __name__ == "__main__":
    main() 