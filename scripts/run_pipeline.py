"""
Main Pipeline - Orchestrates drift detection and response
========================================================
Run this script on a schedule (cron, GitHub Actions, etc.)

Usage:
    python run_pipeline.py                    # Single run
    python run_pipeline.py --continuous 60    # Run every 60 minutes
    python run_pipeline.py --help             # Show help
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drift_monitor import DriftMonitor
from src.triggers import TriggerActions


def print_banner(text: str):
    """Print a formatted banner."""
    print("\n" + "="*70)
    print(f"{text:^70}")
    print("="*70 + "\n")


def run_pipeline() -> bool:
    """
    Main pipeline execution.
    
    Returns:
        True if successful, False otherwise
    """
    
    print_banner("DRIFT-AWARE RETRAINING PIPELINE")
    print(f"{'Run started: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^70}\n")
    
    # Initialize components
    try:
        print("Initializing components...")
        monitor = DriftMonitor()
        triggers = TriggerActions()
        print("✓ Initialization complete\n")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    try:
        # === PHASE 1: DRIFT DETECTION ===
        print_banner("PHASE 1: DRIFT DETECTION")
        results = monitor.run_full_check()
        
        # === PHASE 2: AUTOMATED RESPONSE ===
        if results["drifts_detected"]:
            print_banner("PHASE 2: AUTOMATED RESPONSE")
            response = triggers.execute_drift_response(results["drifts_detected"])
            
            # Print success summary
            print_banner("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print(f"Drifts detected: {', '.join(results['drifts_detected'])}")
            print(f"Actions taken: {', '.join(response['actions_taken'])}")
            print(f"\n{'='*70}\n")
        else:
            # No drift detected
            print_banner("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("No drift detected - system is healthy")
            print(f"\n{'='*70}\n")
        
        return True
        
    except Exception as e:
        print_banner("❌ PIPELINE FAILED")
        print(f"Error: {e}\n")
        
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        print()
        
        return False


def run_continuous(interval_minutes: int = 60):
    """
    Run pipeline continuously with specified interval.
    
    Args:
        interval_minutes: Time between runs
    """
    print_banner("CONTINUOUS MONITORING MODE")
    print(f"Interval: {interval_minutes} minutes")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 Press Ctrl+C to stop\n")
    print("="*70 + "\n")
    
    run_count = 0
    
    try:
        while True:
            run_count += 1
            
            print(f"\n{'🔄 RUN #{run_count}':^70}\n")
            
            # Run pipeline
            success = run_pipeline()
            
            if not success:
                print(f"⚠️ Run #{run_count} failed, but continuing...")
            
            # Calculate next run time
            next_run = datetime.now()
            next_run = next_run.replace(
                minute=(next_run.minute + interval_minutes) % 60,
                second=0,
                microsecond=0
            )
            
            print(f"\n{'='*70}")
            print(f"{'😴 SLEEPING':^70}")
            print(f"{'='*70}")
            print(f"Next run at: {next_run.strftime('%H:%M:%S')}")
            print(f"Sleeping for {interval_minutes} minutes...")
            print(f"{'='*70}\n")
            
            time.sleep(interval_minutes * 60)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print(f"{'👋 CONTINUOUS MONITORING STOPPED':^70}")
        print("="*70)
        print(f"Total runs completed: {run_count}")
        print(f"Stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")


def show_help():
    """Display help information."""
    help_text = """
╔═══════════════════════════════════════════════════════════════╗
║         Drift-Aware Retraining Pipeline - CLI Help           ║
╚═══════════════════════════════════════════════════════════════╝

USAGE:
    python run_pipeline.py [OPTIONS]

OPTIONS:
    (none)              Run pipeline once and exit
    --continuous [MIN]  Run continuously with interval
                        Default: 60 minutes
    --help, -h          Show this help message

EXAMPLES:
    python run_pipeline.py
        → Run once and check for drift
    
    python run_pipeline.py --continuous
        → Run every 60 minutes (default)
    
    python run_pipeline.py --continuous 30
        → Run every 30 minutes
    
    python run_pipeline.py --help
        → Show this help

WHAT IT DOES:
    1. Checks for 3 types of drift:
       • Embedding drift (data distribution changes)
       • Behavior drift (refusal/toxicity spikes)
       • Accuracy drift (performance degradation)
    
    2. If drift detected, automatically:
       • Re-indexes documents
       • Refreshes retrieval system
       • Fine-tunes model
    
    3. Logs all events to Supabase

MONITORING:
    View metrics at: http://localhost:8000/metrics
    (Run: python src/metrics_exporter.py)

LOGS:
    Check logs/ directory for detailed output

ENVIRONMENT:
    Make sure .env file is configured with:
    • SUPABASE_URL
    • SUPABASE_KEY
    • OPENROUTER_API_KEY

═══════════════════════════════════════════════════════════════
"""
    print(help_text)


def main():
    """Main entry point."""
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("\n❌ ERROR: .env file not found!")
        print("Please create .env file with required credentials.")
        print("See README.md for setup instructions.\n")
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h', 'help']:
            show_help()
            sys.exit(0)
        
        elif arg == '--continuous':
            # Get interval from next argument or use default
            interval = 60
            if len(sys.argv) > 2:
                try:
                    interval = int(sys.argv[2])
                    if interval < 1:
                        print("❌ ERROR: Interval must be at least 1 minute")
                        sys.exit(1)
                except ValueError:
                    print(f"❌ ERROR: Invalid interval '{sys.argv[2]}'. Must be a number.")
                    sys.exit(1)
            
            run_continuous(interval_minutes=interval)
        
        else:
            print(f"❌ ERROR: Unknown argument '{sys.argv[1]}'")
            print("Use --help for usage information")
            sys.exit(1)
    
    else:
        # Single run
        success = run_pipeline()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
