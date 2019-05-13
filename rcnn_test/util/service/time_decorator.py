import time

from util.service.config import Config


def spend_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        if Config.SPEND_TIME_VISIBLE:
            print('{} 方法执行时间\t{}'.format(func.__name__, time.time() - start))
        return result
    return wrapper
