#!/usr/bin/env python3
"""
Script to start the visualization server for attribution graphs.
Run this script to view interactive attribution graphs in your browser.
"""

import sys
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Start visualization server for attribution graphs")
    parser.add_argument("--data-dir", type=str, default="safety_analysis_results/dashboard/graph_files",
                       help="Directory containing graph files (default: safety_analysis_results/dashboard/graph_files)")
    parser.add_argument("--port", type=int, default=8032,
                       help="Port to serve on (default: 8032)")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Host to bind to (default: localhost)")
    
    args = parser.parse_args()
    
    # Check if data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist!")
        print("Please run the safety analysis first to generate graph files.")
        sys.exit(1)
    
    # Check if there are any graph files
    graph_files = list(data_dir.glob("*.json"))
    if not graph_files:
        print(f"Error: No graph files found in '{data_dir}'!")
        print("Please run the safety analysis first to generate graph files.")
        sys.exit(1)
    
    print(f"Found {len(graph_files)} graph files in {data_dir}")
    
    try:
        from circuit_tracer.frontend.local_server import serve
        
        print(f"Starting visualization server...")
        print(f"  Data directory: {data_dir}")
        print(f"  URL: http://{args.host}:{args.port}")
        print(f"  Press Ctrl+C to stop the server")
        print()
        
        # Start the server
        server = serve(data_dir, port=args.port)
        
        # Keep the server running
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping server...")
            server.stop()
            print("Server stopped.")
            
    except ImportError:
        print("Error: Could not import circuit_tracer. Please install it first:")
        print("  pip install circuit-tracer")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 