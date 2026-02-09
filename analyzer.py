import re

def extract_metric(text, pattern):
    """
    Extracts a floating point value using a regex pattern.
    """
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def analyze_report(text):
    """
    Parses text and extracts medical metrics.
    Returns a dictionary of findings in a standard format.
    """
    findings = {}

    # Define regex patterns for common metrics
    patterns = {
        "Hemoglobin": r"Hemoglobin[:\s]+([\d\.]+)",
        "Total Cholesterol": r"Total Cholesterol[:\s]+([\d\.]+)",
        "LDL Cholesterol": r"LDL Cholesterol[:\s]+([\d\.]+)",
        "HDL Cholesterol": r"HDL Cholesterol[:\s]+([\d\.]+)",
        "Triglycerides": r"Triglycerides[:\s]+([\d\.]+)",
        "Fasting Blood Sugar": r"Fasting Blood Sugar[:\s]+([\d\.]+)",
        "Post Prandial Blood Sugar": r"Post Prandial Blood Sugar[:\s]+([\d\.]+)",
        "HbA1c": r"HbA1c[:\s]+([\d\.]+)",
    }

    # Blood Pressure requires special handling for sys/dia
    bp_match = re.search(r"(\d{2,3})[\s/]+(\d{2,3})\s*mmHg", text)
    if bp_match:
        findings["Blood Pressure"] = (f"{bp_match.group(1)}/{bp_match.group(2)}", "High" if int(bp_match.group(1)) > 120 or int(bp_match.group(2)) > 80 else "Normal")
    
    for name, pattern in patterns.items():
        val = extract_metric(text, pattern)
        if val is not None:
            # Simple reference check (this should ideally be driven by a config or DB)
            status = "Normal"
            if name == "Hemoglobin" and (val < 13 or val > 17): status = "Abnormal" # Male range rough
            elif name == "Total Cholesterol" and val > 200: status = "High"
            elif name == "LDL Cholesterol" and val > 100: status = "High"
            elif name == "HDL Cholesterol" and val < 40: status = "Low"
            elif name == "Triglycerides" and val > 150: status = "High"
            elif name == "Fasting Blood Sugar" and val > 100: status = "High"
            elif name == "HbA1c" and val > 5.7: status = "High"
            
            findings[name] = (val, status)

    return findings
