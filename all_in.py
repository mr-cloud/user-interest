import preprocessing_photos
import modeling_k_means
import preprocessing_user_preferences
import recommend_for_each_user

import datetime
import os


with open(os.path.join(preprocessing_photos.DATA_HOUSE_PATH, 'all-in.log'), 'w') as logger:
    logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Started preprocessing_photos'))
    preprocessing_photos.main(logger)
    logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Started modeling_k_means'))
    modeling_k_means.main(logger)
    logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Started preprocessing_user_preferences'))
    preprocessing_user_preferences.main(logger)
    logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Started recommend_for_each_user'))
    recommend_for_each_user.main('v0.9.0', logger)
    logger.write('{}: {}\n'.format(datetime.datetime.now(), 'Finished'))


