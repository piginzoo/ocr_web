# -*- coding: utf-8 -*-
"""
    说明：
"""
import json
import os

__config_path = os.path.join(os.path.abspath(''), "config/config.json")
with open(__config_path, "r") as fr:
    _data = json.loads(fr.read())

def get(key=None):
    if key == None:
        return _data
    return _data.get(key)
