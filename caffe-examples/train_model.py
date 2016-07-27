
#
# Train 'caffe-net' model to custom train set
#

# Python
import os
import sys
import subprocess

### Setup pathes
CAFFE_TOOLS_PATH = "/Users/vfomin/Documents/ML/caffe-master_42cd785/build/tools/"

assert os.path.exists(CAFFE_TOOLS_PATH), "Caffe tools path is not found"
CAFFE_EXEC_PATH = os.path.join(CAFFE_TOOLS_PATH, 'caffe')
assert os.path.exists(CAFFE_EXEC_PATH), "caffe executable tool is not found"


### Test launch caffe
# program = [CAFFE_EXEC_PATH, 'device_query', '-gpu', '0']
# ret = subprocess.Popen(program)
# ret.wait()
# assert ret.returncode == 0, "Caffe execution is failed"



### Finetune model
program = [CAFFE_EXEC_PATH,
           'train',
           '-solver', 'models/train_solver.prototxt']

if len(sys.argv) > 1:
    program.extend(sys.argv[1:])

ret = subprocess.Popen(program)
ret.wait()
assert ret.returncode == 0, "Caffe execution is failed"
