import os

# Auto-detect which machine we're on
if os.path.exists("/home/karthiksunil/work/code/cv/CuisineProject"):
    # Your local PC
    DATA_ROOT = (
        "/home/karthiksunil/work/code/cv/CuisineProject/Khana Dataset/extracted/khana"
    )
    CHECKPOINT_DIR = "./checkpoints"
else:
    # Lab machine (you'll update this tomorrow)
    DATA_ROOT = "~/data/khana"
    CHECKPOINT_DIR = "~/checkpoints"
