import pickle
import os

def print_keys(filepath, num_keys=5):
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"--- Keys from {filepath} ---")
        if isinstance(data, dict):
            if "omics_dict" in filepath: # Special handling for omics_dict.pkl
                for omics_type, omics_data in data.items():
                    print(f"  Omics Type: {omics_type}")
                    if isinstance(omics_data, dict):
                        keys = list(omics_data.keys())
                        for i in range(min(num_keys, len(keys))):
                            print(f"    {keys[i]}")
                    else:
                        print(f"    Data for {omics_type} is not a dictionary (type: {type(omics_data)})")
            else:
                keys = list(data.keys())
                for i in range(min(num_keys, len(keys))):
                    print(keys[i])
        else:
            print(f"Data is not a dictionary (type: {type(data)})")
        print("-" * 30)
    except Exception as e:
        print(f"Error loading or inspecting {filepath}: {e}")

# Example usage - replace with your actual file paths
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for pkl_file in sys.argv[1:]:
            print_keys(pkl_file)
    else:
        print("Usage: python inspect_pkl_keys.py <pkl_file1> [pkl_file2] ...")
        print("Example:")
        print("  python inspect_pkl_keys.py wsi_feature/wsi_features_ACC.pkl")
        print("  python inspect_pkl_keys.py label_feature/label_dict_ACC.pkl")
        print("  python inspect_pkl_keys.py omics_feature/omics_dict.pkl")
