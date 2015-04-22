from numpy import sqrt


def parallel(obj):
    global pyspark, sc
    if isinstance(obj, pyspark.rdd.RDD):
        return obj
    else:
        return sc.parallelize(obj)


def rdd_sum(rdd):
    return parallel(rdd).reduce(lambda x, y: x + y)


def pair_rdd_sum(pair_rdd):
    return parallel(pair_rdd).reduceByKey(lambda x, y: x + y)


def rdd_mean(rdd, count=None):
    if count is None:
        sum_and_count = parallel(rdd).map(lambda x: (x, 1))\
            .reduce(lambda tuple1, tuple2: tuple(i + j for i, j in zip(tuple1, tuple2)))
        return sum_and_count[0] / sum_and_count[1]
    else:
        return parallel(rdd).reduce(lambda x, y: x + y) / count


def pair_rdd_mean(pair_rdd, count_by_key=None):
    if count_by_key is None:
        sums_and_counts = parallel(pair_rdd).mapValues(lambda x: (x, 1))\
            .reduceByKey(lambda tuple1, tuple2: tuple(i + j for i, j in zip(tuple1, tuple2)))
    else:
        sum_by_key = pair_rdd_sum(pair_rdd)
        count_by_key = parallel(count_by_key)
        sums_and_counts = sum_by_key.cogroup(count_by_key)
    return sums_and_counts.mapValues(lambda sum_and_count: sum_and_count[0] / sum_and_count[1])


def rdd_variance(rdd, count=None, mean=None):
    rdd = parallel(rdd)
    if count is None:
        count = rdd.count()
    if count < 2:
        return 0.0
    else:
        if mean is None:
            mean = rdd_mean(rdd, count=count)
        return rdd.map(lambda x: (x - mean) ** 2).reduce(lambda x, y: x + y) / (count - 1)


def pair_rdd_variance(pair_rdd, count_by_key=None, mean_by_key=None):
    pair_rdd = parallel(pair_rdd)
    if count_by_key is None:
        count_by_key = parallel(pair_rdd.countByKey())
    else:
        count_by_key = parallel(count_by_key)
    if mean_by_key is None:
        mean_by_key = pair_rdd_mean(pair_rdd, count_by_key=count_by_key)
    else:
        mean_by_key = parallel(mean_by_key)
    return pair_rdd.groupWith(count_by_key, mean_by_key)\
        .mapValues(lambda vector_and_count_and_mean:
                   rdd_variance(vector_and_count_and_mean[0],
                                count=vector_and_count_and_mean[1], mean=vector_and_count_and_mean[2]))


def rdd_standard_deviation(rdd, count=None, mean=None):
    return sqrt(rdd_variance(rdd, count=count, mean=mean))


def pair_rdd_standard_deviation(pair_rdd, count_by_key=None, mean_by_key=None):
    return pair_rdd_variance(pair_rdd, count_by_key=count_by_key, mean_by_key=mean_by_key)\
        .mapValues(lambda variance: sqrt(variance))


def rdd_normalize_subtract_mean_divide_standard_deviation(rdd, count=None, mean=None, standard_deviation=None):
    rdd = parallel(rdd)
    if count is None:
        count = rdd.count()
    if mean is None:
        mean = rdd_mean(rdd, count=count)
    if standard_deviation is None:
        standard_deviation = rdd_standard_deviation(rdd, count=count, mean=mean)
    return rdd.map(lambda x: (x - mean) / standard_deviation)


def pair_rdd_normalize_subtract_mean_divide_standard_deviation(pair_rdd, count_by_key=None, mean_by_key=None,
                                                               standard_deviation_by_key=None):
    pair_rdd = parallel(pair_rdd)
    if count_by_key is None:
        count_by_key = parallel(pair_rdd.countByKey)
    else:
        count_by_key = parallel(count_by_key)
    if mean_by_key is None:
        mean_by_key = pair_rdd_mean(pair_rdd, count_by_key=count_by_key)
    else:
        mean_by_key = parallel(mean_by_key)
    if standard_deviation_by_key is None:
        standard_deviation_by_key = pair_rdd_standard_deviation(pair_rdd, count_by_key=count_by_key,
                                                                mean_by_key=mean_by_key)
    else:
        standard_deviation_by_key = parallel(standard_deviation_by_key)
    return pair_rdd.groupWith(count_by_key, mean_by_key, standard_deviation_by_key)\
        .mapValues(lambda vector_and_count_and_mean_and_standard_deviation:
                   rdd_normalize_subtract_mean_divide_standard_deviation(
                       vector_and_count_and_mean_and_standard_deviation[0],
                       count=vector_and_count_and_mean_and_standard_deviation[1],
                       mean=vector_and_count_and_mean_and_standard_deviation[2],
                       standard_deviation=vector_and_count_and_mean_and_standard_deviation[3]))