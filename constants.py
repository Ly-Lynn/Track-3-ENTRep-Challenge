# Model constants
IMAGE_SIZE = 224
CLIP_MAX_LENGTH = 77
DEFAULT_TEMPERATURE = 0.07
DEFAULT_FEATURE_DIM = 768

# Training constants
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BETAS = (0.9, 0.98)
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0

# EMA constants
DEFAULT_EMA_DECAY = 0.995

# Early stopping
DEFAULT_PATIENCE = 5

# Scheduler constants
DEFAULT_START_FACTOR = 0.1

# Image normalization (CLIP standard)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# CLIP model name
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# File extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

# Metrics
DEFAULT_K_VALUES = [1, 5, 10]

# Logging
LOG_FREQUENCY = 10  # Log every N batches 