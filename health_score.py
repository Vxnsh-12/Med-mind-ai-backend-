def analyze_value(value, low, high):
    """
    Categorizes a value based on reference range.
    Returns: 'Normal', 'Low', 'High'
    """
    if value < low:
        return "Low"
    elif value > high:
        return "High"
    return "Normal"

def calculate_score(metrics):
    """
    Calculates a health score (0-100) based on deviations.
    metrics: dict of {metric_name: (value, status)}
    
    Base Score: 100
    Deductions:
    - High/Low critical metric (e.g., BP, Sugar): -10
    - High/Low moderate metric (e.g., Cholesterol): -5
    """
    score = 100
    details = []

    # Weighted penalties
    penalties = {
        "Blood Pressure": 15,
        "Fasting Blood Sugar": 15,
        "HbA1c": 20,
        "Total Cholesterol": 10,
        "LDL Cholesterol": 10,
        "Hemoglobin": 10,
        "Triglycerides": 5
    }

    for metric, (value, status) in metrics.items():
        if status != "Normal":
            penalty = penalties.get(metric, 5)
            score -= penalty
            details.append(f"{metric} is {status} ({value}). Deduction: -{penalty}")

    return max(0, score), details
