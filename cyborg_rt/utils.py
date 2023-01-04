"""
utils.py - A CYBORG-SAL-Collab file
Copyright (C) 2021  Zach Carmichael
"""
import logging
import os
import sys
import math


def get_logger(name):
    logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get('CYBORG_SAL_LOG_LEVEL', 'INFO'))
    return logger


def dict_union(*dicts):
    if sys.version_info.major >= 3 and sys.version_info.minor >= 9:
        union = dicts[0]
        for d in dicts[1:]:
            union |= d
    else:
        union = dicts[0].copy()
        for d in dicts[1:]:
            union.update(d)
    return union


def num_cpus():
    if 'JOB_ID' in os.environ:
        # assume SGE environment
        base_err = ('Inferred that you are in an SGE environment (because '
                    f'$JOB_ID is set as {os.environ["JOB_ID"]}) but $NSLOTS '
                    f'is not ')
        try:
            return int(os.environ['NSLOTS'])
        except KeyError:
            raise RuntimeError(base_err + 'set!')
        except ValueError:
            raise RuntimeError(base_err + f'an int ({os.environ["NSLOTS"]})!')
    else:
        # assume no scheduler (resource allocation)
        return os.cpu_count()


if hasattr(math, 'prod'):  # available in 3.8+
    prod = math.prod
else:  # functionally equivalent w/o positional argument checking
    """
    >>> %timeit reduce(mul, values)
    180 µs ± 2.15 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    >>> %timeit math.prod(values)
    133 µs ± 1.57 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    >>> math.prod(values) == reduce(mul, values)
    True
    """
    import operator
    from functools import reduce


    def prod(iterable, start=1):
        return reduce(operator.mul, iterable, start)


def requires_human_annotations(C):
    loss_split = C.LOSS.upper().split('+')
    # CYBORG loss or SAL loss with human annotations as salience maps
    return ('CYBORG' in loss_split or
            ('SAL' in loss_split and C.SAL_LOSS_METHOD == 'annotations'))
