# Medical Report Analysis

A Python tool to analyze medical reports and provide a health score.

## Setup
1. Standard Python 3 installation.
2. (Optional) Convert PDF reports to text or install `pypdf`:
   ```bash
   pip install pypdf
   ```

## Usage
Run the main script and provide the path to your report file (Text or PDF).

```bash
python main.py [path_to_file]
```

### Example
```bash
python main.py sample_report.txt
```

## Features
- Extracts key metrics: BP, Cholesterol, Sugar, Hemoglobin.
- Calculates a health score (0-100).
- Highlights abnormal values.
