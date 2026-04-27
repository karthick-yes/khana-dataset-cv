import os


def get_paths():

    lab_path = os.path.expanduser("~/data/khana")
    if os.path.exists(lab_path):
        return {
            "DATA_ROOT": lab_path,
            "CHECKPOINT_DIR": os.path.expanduser("~/data/checkpoints"),
            "LABELS_PATH": os.path.expanduser("~/data/labels.txt"),
            "TAXONOMY_PATH": os.path.expanduser("~/data/taxonomy.csv"),
            "MACHINE": "lab_machine",
        }

    os.makedirs(os.path.expanduser("~/data/checkpoints"), exist_ok=True)
    return {
        "DATA_ROOT": os.path.expanduser("~/data/khana"),
        "CHECKPOINT_DIR": os.path.expanduser("~/data/checkpoints"),
        "LABELS_PATH": os.path.expanduser("~/data/labels.txt"),
        "TAXONOMY_PATH": os.path.expanduser("~/data/taxonomy.csv"),
        "MACHINE": "fallback",
    }
