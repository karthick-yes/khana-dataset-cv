import os

def get_paths():
    """
    Auto-detect machine and return correct paths.
    Works on: local PC, lab machine (Pop!_OS), WSL
    """
    
    # Check if we're on your local PC
    pc_path = "/home/karthiksunil/work/code/cv/CuisineProject/Khana Dataset/extracted/khana"
    if os.path.exists(pc_path):
        return {
            "DATA_ROOT": pc_path,
            "CHECKPOINT_DIR": "/home/karthiksunil/work/code/cv/CuisineProject/checkpoints",
            "LABELS_PATH": "/home/karthiksunil/work/code/cv/CuisineProject/Khana Dataset/labels.txt",
            "TAXONOMY_PATH": "/home/karthiksunil/work/code/cv/CuisineProject/Khana Dataset/taxonomy.csv",
            "MACHINE": "local_pc"
        }
    
    # Check if we're on lab machine (Pop!_OS)
    lab_path = os.path.expanduser("~/data/khana")
    if os.path.exists(lab_path):
        return {
            "DATA_ROOT": lab_path,
            "CHECKPOINT_DIR": os.path.expanduser("~/checkpoints"),
            "LABELS_PATH": os.path.expanduser("~/data/labels.txt"),
            "TAXONOMY_PATH": os.path.expanduser("~/data/taxonomy.csv"),
            "MACHINE": "lab_machine"
        }
    
    # Fallback: assume lab machine, create dirs if needed
    lab_path = os.path.expanduser("~/data/khana")
    os.makedirs(lab_path, exist_ok=True)
    os.makedirs(os.path.expanduser("~/checkpoints"), exist_ok=True)
    return {
        "DATA_ROOT": lab_path,
        "CHECKPOINT_DIR": os.path.expanduser("~/checkpoints"),
        "LABELS_PATH": os.path.expanduser("~/data/labels.txt"),
        "TAXONOMY_PATH": os.path.expanduser("~/data/taxonomy.csv"),
        "MACHINE": "lab_machine_fallback"
    }

# Usage in other files:
# from config import get_paths
# paths = get_paths()
# DATA_ROOT = paths["DATA_ROOT"]