from celery_package.tasks import one


def notify():
    result = one.apply_async(args=[1, 3])
    return result


if __name__ == '__main__':
    result = notify()
    print(result)
    print(result.status)
    print(result.id)
    # testFl = [1, 2, 3, 4,5,6]
    # start = time.time()
    # for fn in testFl:
    #     run(fn)
    # e1 = time.time()
    # print({'单进程': (e1 - start)})
    # pool = Pool(6)
    # r = pool.map(run, testFl)
    # # 关闭不再接收线程
    # pool.close()
    # # 主进程阻塞等待所有子进程执行完毕
    # pool.join()
    # e2 = time.time()
    #
    # print('多进程 ' + '\t' + str((e2 - e1)))
    # print(r)
