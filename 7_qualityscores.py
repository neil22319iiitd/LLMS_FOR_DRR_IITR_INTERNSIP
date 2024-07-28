import csv
from sklearn.metrics import precision_score, recall_score, f1_score

def load_data(file_path):
    operational_status = []
    impact_labels = []
    severity = []
    
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            operational_status.append(row['Operational Status'])
            impact_labels.append(row['Impact Label'])
            severity.append(row['Severity'])
    
    return operational_status, impact_labels, severity

# Load the ground truth and actual results
ground_truth_file = "week7_ground_truth.csv"
actual_results_file = "week7_llama_2_13B_2.csv"

ground_truth_operational, ground_truth_impact, ground_truth_severity = load_data(ground_truth_file)
actual_operational, actual_impact, actual_severity = load_data(actual_results_file)

# Calculate precision, recall, and F1 score for operational status
op_precision = precision_score(ground_truth_operational, actual_operational, average='weighted', zero_division=0)
op_recall = recall_score(ground_truth_operational, actual_operational, average='weighted', zero_division=0)
op_f1 = f1_score(ground_truth_operational, actual_operational, average='weighted', zero_division=0)

print("Operational Status:")
print(f"Precision: {op_precision:.2f}")
print(f"Recall: {op_recall:.2f}")
print(f"F1 Score: {op_f1:.2f}")

# Calculate precision, recall, and F1 score for impact labels
impact_precision = precision_score(ground_truth_impact, actual_impact, average='weighted', zero_division=0)
impact_recall = recall_score(ground_truth_impact, actual_impact, average='weighted', zero_division=0)
impact_f1 = f1_score(ground_truth_impact, actual_impact, average='weighted', zero_division=0)

print("Impact Labels:")
print(f"Precision: {impact_precision:.2f}")
print(f"Recall: {impact_recall:.2f}")
print(f"F1 Score: {impact_f1:.2f}")

# Calculate precision, recall, and F1 score for severity
severity_precision = precision_score(ground_truth_severity, actual_severity, average='weighted', zero_division=0)
severity_recall = recall_score(ground_truth_severity, actual_severity, average='weighted', zero_division=0)
severity_f1 = f1_score(ground_truth_severity, actual_severity, average='weighted', zero_division=0)

print("Severity:")
print(f"Precision: {severity_precision:.2f}")
print(f"Recall: {severity_recall:.2f}")
print(f"F1 Score: {severity_f1:.2f}")
