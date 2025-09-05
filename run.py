#!/usr/bin/env python3
"""
Quick run script for CramAI
"""

import os
import sys
import subprocess

def check_env():
    """Check if .env file exists and has API key"""
    if not os.path.exists(".env"):
        print("❌ .env file not found")
        print("Please run: python setup.py")
        return False
    
    with open(".env", "r") as f:
        content = f.read()
        if "your_openai_api_key_here" in content:
            print("⚠️  Please edit .env file and add your OpenAI API key")
            return False
    
    return True

def main():
    """Main run function"""
    print("🧠 Starting CramAI...")
    
    if not check_env():
        sys.exit(1)
    
    print("🚀 Launching Streamlit app...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 CramAI stopped")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()
