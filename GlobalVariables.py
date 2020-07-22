import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'result')
WEIGHT_DIR = os.path.join(RESULT_DIR, 'weight')
DATA_DIR = os.path.join(RESULT_DIR, 'data')

DEFAULT_CONF_DIR = os.path.join(BASE_DIR, 'conf')
CONF_CLF_DIR = os.path.join(DEFAULT_CONF_DIR, 'Classification')
CONF_SEG_DIR = os.path.join(DEFAULT_CONF_DIR, 'Segmentation')
CONF_AUG_DIR = os.path.join(DEFAULT_CONF_DIR, 'Augmentation')
CONF_REALTIME_DIR = os.path.join(DEFAULT_CONF_DIR, 'RealTime')

DEFAULT_LOG_DIR = os.path.join(RESULT_DIR, 'log')
DEFAULT_LOGGER_PATH = os.path.join(DEFAULT_LOG_DIR, 'common.log')
DEFAULT_LOGGER = 'common'

TENSORBOARD_DIR = os.path.join(RESULT_DIR, 'tensorboard')
CSV_DIR = os.path.join(RESULT_DIR, 'csv')