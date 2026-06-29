#!/usr/bin/env python
"""
Run the Drone AI Simulation Dashboard.

Usage:
    python run_simulation.py [--port PORT] [--debug] [--fastapi]

This will start a web server and open a browser with the drone simulation dashboard.
"""

import argparse
import webbrowser
import threading
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def open_browser(port):
    """Open browser after a short delay."""
    time.sleep(1.5)
    webbrowser.open(f'http://localhost:{port}')


def run_flask(host, port, debug, no_browser):
    """Run Flask server."""
    try:
        from flask import Flask
        from flask_socketio import SocketIO
    except ImportError:
        print("\nMissing Flask dependencies! Please install them first:")
        print("  pip install flask flask-socketio")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # Open browser in background thread
    if not no_browser:
        browser_thread = threading.Thread(target=open_browser, args=(port,), daemon=True)
        browser_thread.start()

    # Start server
    from simulation.web_server import run_server
    run_server(host=host, port=port, debug=debug)


def run_fastapi(host, port, debug, no_browser):
    """Run FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("\nMissing FastAPI dependencies! Please install them first:")
        print("  pip install fastapi uvicorn[standard] websockets pydantic")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # Open browser in background thread
    if not no_browser:
        browser_thread = threading.Thread(target=open_browser, args=(port,), daemon=True)
        browser_thread.start()

    print(f"\n{'='*60}")
    print(f"  Drone Simulation Dashboard (FastAPI)")
    print(f"  Open in browser: http://localhost:{port}")
    print(f"  API Documentation: http://localhost:{port}/docs")
    print(f"{'='*60}\n")

    # Start FastAPI server
    uvicorn.run(
        "simulation.fastapi_server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning",
    )


def main():
    parser = argparse.ArgumentParser(description='Run the Drone AI Simulation Dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--fastapi', action='store_true', help='Use FastAPI instead of Flask (recommended for new features)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()

    if args.fastapi:
        run_fastapi(args.host, args.port, args.debug, args.no_browser)
    else:
        run_flask(args.host, args.port, args.debug, args.no_browser)


if __name__ == '__main__':
    main()
