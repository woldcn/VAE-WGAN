# Author：woldcn
# Create Time：2022/10/4 18:03
# Description：

import json
from args import predicotr_args as cf

with open(cf.test_path, 'r') as f:
    json_data = json.load(f)
    print(len(json_data))