
#
# Finetune 'caffe-net' model to custom train set
#

# Python
import os
import sys
import subprocess

### Setup pathes
CAFFE_TOOLS_PATH = "/Users/vfomin/Documents/ML/caffe-master_42cd785/build/tools/"

# Caffe net weights
CAFFE_MODEL_WEIGHTS_PATH = "/Users/vfomin/Documents/ML/caffe-master_42cd785/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
# Googlenet weights
#CAFFE_MODEL_WEIGHTS_PATH = "/Users/vfomin/Documents/ML/caffe-master_42cd785/models/bvlc_googlenet/bvlc_googlenet.caffemodel"
# ResNet-152 weights
#CAFFE_MODEL_WEIGHTS_PATH = "/Users/vfomin/Documents/ML/caffe-master_42cd785/models/ResNet/ResNet-152-model.caffemodel"
# ResNet-110 weights
#CAFFE_MODEL_WEIGHTS_PATH = "/Users/vfomin/Documents/ML/caffe-master_42cd785/models/ResNet/ResNet-101-model.caffemodel"
# ResNet-50 weights
#CAFFE_MODEL_WEIGHTS_PATH = "/Users/vfomin/Documents/ML/caffe-master_42cd785/models/ResNet/ResNet-50-model.caffemodel"


assert os.path.exists(CAFFE_TOOLS_PATH), "Caffe tools path is not found"
CAFFE_EXEC_PATH = os.path.join(CAFFE_TOOLS_PATH, 'caffe')
assert os.path.exists(CAFFE_EXEC_PATH), "caffe executable tool is not found"
assert os.path.exists(CAFFE_MODEL_WEIGHTS_PATH), "Caffe model is not found"



### Test launch caffe
# program = [CAFFE_EXEC_PATH, 'device_query', '-gpu', '0']
# ret = subprocess.Popen(program)
# ret.wait()
# assert ret.returncode == 0, "Caffe execution is failed"


### Finetune model
program = [CAFFE_EXEC_PATH,
           'train',
           '-solver', 'models/finetune_solver.prototxt']

if len(sys.argv) > 1:
    program.extend(sys.argv[1:])

if not ('-weights' in " ".join(program)):
    program.extend(['-weights', CAFFE_MODEL_WEIGHTS_PATH])

ret = subprocess.Popen(program)
ret.wait()
assert ret.returncode == 0, "Caffe execution is failed"
