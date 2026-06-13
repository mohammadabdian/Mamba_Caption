import os

# ----------------------------
# BASE DIRECTORY
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# DATA PATHS
# ----------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "flickr30k_images")  # images folder
CAPTIONS_DIR = os.path.join(DATA_DIR, "captions_csv")    # captions CSV folder

# ----------------------------
# FEATURE PATHS
# ----------------------------
FEATURES_DIR = os.path.join(BASE_DIR, "features")
TRAIN_FEATURES_DIR = os.path.join(FEATURES_DIR, "train")
VAL_FEATURES_DIR = os.path.join(FEATURES_DIR, "val")
TEST_FEATURES_DIR = os.path.join(FEATURES_DIR, "test")

# ----------------------------
# MODEL / CHECKPOINT PATHS
# ----------------------------
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# ----------------------------
# UTILITY FUNCTION
# ----------------------------
def make_dirs():
    """Create all required directories if they don't exist"""
    for path in [
        DATA_DIR, IMAGES_DIR, CAPTIONS_DIR,
        FEATURES_DIR, TRAIN_FEATURES_DIR, VAL_FEATURES_DIR, TEST_FEATURES_DIR,
        MODEL_DIR, CHECKPOINT_DIR
    ]:
        os.makedirs(path, exist_ok=True)