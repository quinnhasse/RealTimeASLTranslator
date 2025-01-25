import pandas as pd

file_path = "notebook/data"
def load_labels(file_path):
    
    # Load labels from a CSV file
    data = pd.read_csv(file_path)
    return data