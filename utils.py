import os


log_dir = 'datahouse'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
logger = open(os.path.join(log_dir, 'all-in.log'), 'w')