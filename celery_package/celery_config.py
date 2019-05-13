Config = {
    'BROKER_URL': 'redis://127.0.0.1:6379/0',
    'CELERY_RESULT_BACKEND': 'redis://127.0.0.1:6379/0',
    'CELERY_TASK_SERIALIZER': 'json',
    'CELERY_RESULT_SERIALIZER': 'json',
    'CELERY_ACCEPT_CONTENT': ['json'],
    # 长时间运行celery可能会导致内存泄露，需设置worker任务最大值
    'CELERY_MAX_TASKS_PER_CHILD': 100,
    # 并发数
    'CELERY_CONCURRENCY': 1,
    # 引入tasks注意路径
    'CELERY_IMPORTS': ('celery_package.tasks',),
    # 任务队列
    # CELERY_DEFAULT_QUEUE:
    'CELERY_TASK_RESULT_EXPIRES': 60 * 60 * 1,
}
