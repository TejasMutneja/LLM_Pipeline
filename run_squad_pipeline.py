#!/usr/bin/env python3
"""
SQuAD MCQ Pipeline Runner

This script runs the complete SQuAD to MCQ conversion and evaluation pipeline:
1. Converts SQuAD questions to MCQ format
2. Runs the debate pipeline on the converted questions
3. Displays results
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        
        # Print stdout if available
        if result.stdout.strip():
            print(result.stdout)
            
        elapsed_time = time.time() - start_time
        print(f"\n✅ {description} completed successfully in {elapsed_time:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed_time:.1f} seconds")
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def main():
    print("🎯 SQuAD MCQ Pipeline Runner")
    print("This script will:")
    print("1. Convert SQuAD questions to MCQ format")
    print("2. Run the debate pipeline on converted questions")
    print("3. Generate evaluation results")
    
    # Check if required files exist
    required_files = [
        "squad_to_mcq_converter.py",
        "pipeline.py",
        ".env"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all required files are in the current directory.")
        sys.exit(1)
    
    print(f"\n✅ All required files found")
    
    # Step 1: Convert SQuAD to MCQ
    if not run_command("source venv/bin/activate && python squad_to_mcq_converter.py", "Converting SQuAD questions to MCQ format"):
        print("\n❌ Failed to convert SQuAD questions. Please check the error above.")
        sys.exit(1)
    
    # Check if conversion output exists
    if not os.path.exists("squad_mcq_questions.json"):
        print("\n❌ MCQ conversion file not found. Conversion may have failed.")
        sys.exit(1)
    
    print(f"\n✅ SQuAD questions successfully converted to MCQ format")
    
    # Step 2: Run the debate pipeline
    if not run_command("source venv/bin/activate && python pipeline.py", "Running debate pipeline on SQuAD MCQ questions"):
        print("\n❌ Failed to run the debate pipeline. Please check the error above.")
        sys.exit(1)
    
    # Step 3: Check for output files
    output_files = [
        "squad_mcq_debate_loop_eval_results_final.json",
        "squad_mcq_debate_loop_eval_summary_final.csv"
    ]
    
    found_outputs = []
    for file in output_files:
        if os.path.exists(file):
            found_outputs.append(file)
    
    if found_outputs:
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"\n📊 Results saved to:")
        for file in found_outputs:
            size = os.path.getsize(file)
            print(f"  ✅ {file} ({size:,} bytes)")
        
        print(f"\n📋 Summary:")
        print(f"  • SQuAD questions converted to MCQ format")
        print(f"  • Debate pipeline evaluation completed")  
        print(f"  • Results available in JSON and CSV formats")
        
    else:
        print(f"\n⚠️  Pipeline completed but no output files found.")
        print(f"Expected files: {', '.join(output_files)}")

if __name__ == "__main__":
    main()