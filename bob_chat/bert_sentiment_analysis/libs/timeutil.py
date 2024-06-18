#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module is time util

Author:linming(linming@ztgame.com)
Date: 2017/07/06
"""

import os
import sys
import time

def date2timestamp(datestr):
    """
    date: '%Y-%m-%d %H:%M:%S' to timestamp
    """
    try:
        ts = time.mktime(time.strptime(datestr, '%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        raise e
    return int(ts)



def timestamp2date(ts):
    """
    timestamp to date: '%Y-%m-%d %H:%M:%S'
    """
    try:
        datestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
    except Exception as e:
        raise e
    return datestr


if __name__ == '__main__':
    ts = 1499326682
    datestr = timestamp2date(ts)
    print(datestr)
    ts = date2timestamp(datestr)
    print(ts)