import sys
import os
from report_parser import read_file
from analyzer import analyze_report
from health_score import calculate_score

def main():
    print("========================================")
    print("   MedMind - Medical Report Analysis    ")
    print("========================================")
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        print("\nPlease provide the path to your medical report file.")
        filepath = input("File Path: ").strip().strip('"').strip("'")
    
    if not filepath:
        print("No file provided. Exiting.")
        return

    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return

    print("\n[+] Reading file...")
    try:
        text = read_file(filepath)
        # print(f"Debug: Read {len(text)} characters.") 
        
        print("[+] Analyzing data...")
        findings = analyze_report(text)
        
        if not findings:
            print("\n[!] No specific medical metrics could be extracted.")
            print("    Ensure the report is standard text/PDF and contains values like 'Hemoglobin', 'Cholesterol', etc.")
            return

        print("\n" + "-"*40)
        print(f"{'METRIC':<30} | {'VALUE':<10} | {'STATUS'}")
        print("-" * 40)
        
        for metric, (value, status) in findings.items():
            print(f"{metric:<30} | {str(value):<10} | {status}")
            
        print("-" * 40)
        
        score, details = calculate_score(findings)
        
        print("\n" + "="*40)
        print(f"   OVERALL HEALTH SCORE: {score}/100")
        print("=" * 40)
        
        if details:
            print("\nKey Insights & Deductions:")
            for detail in details:
                print(f" -> {detail}")
        else:
            print("\nGreat news! All extracted metrics are within normal ranges.")

    except ImportError as e:
        print(f"\n[Error] Missing Dependency: {e}")
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
