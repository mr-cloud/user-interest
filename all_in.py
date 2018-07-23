# import preprocessing_text_feature_embedding
import  preprocessing_top_word_tfidf
# import preprocessing_photo_face_features

import datetime
from utils import logger


# logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Started ' + 'preprocessing_text_feature_embedding'))
# logger.flush()
# preprocessing_text_feature_embedding.main()
# logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Finished'))
# logger.flush()

# logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Started ' + 'preprocessing_photo_face_features'))
# logger.flush()
# preprocessing_photo_face_features.main()
# logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Finished'))
# logger.flush()

logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Started ' + 'preprocessing_top_word_tfidf'))
logger.flush()
preprocessing_top_word_tfidf.main()
logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Finished'))
logger.flush()

