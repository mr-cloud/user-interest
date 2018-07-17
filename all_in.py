import preprocessing_photo_face_features

import datetime
from utils import logger


logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Started ' + 'preprocessing_photo_face_features'))
preprocessing_photo_face_features.main()
logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Finished'))
