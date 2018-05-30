import preprocessing_photos
import modeling_k_means
import preprocessing_user_preferences
import recommend_for_each_user

import datetime
from utils import  logger


logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Started preprocessing_photos'))
preprocessing_photos.main()
logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Started modeling_k_means'))
modeling_k_means.main()
logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Started preprocessing_user_preferences'))
preprocessing_user_preferences.main()
logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Started recommend_for_each_user'))
recommend_for_each_user.main('v0.9.0')
logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Finished'))


