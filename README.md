# Error Collections!

- - -
### Error List
* #### [﻿RuntimeError: cuda runtime error (710) : device-side assert triggered](#error1)  [▶Blog](https://blog.naver.com/jaeyoon_95/222104626159) 
* #### [﻿RuntimeError: Expected object of scalar type Half but got scalar type Float](#error2)  [▶Blog](https://blog.naver.com/jaeyoon_95/222064412708) 
* #### [﻿AttributeError: module 'tensorflow' has no attribute 'sub'](#error3)  [▶Blog](https://blog.naver.com/jaeyoon_95/222007030881)   
* #### [ModuleNotFoundError: No module named 'sklearn'](#error4)  [▶Blog](https://blog.naver.com/jaeyoon_95/222007026711)   
* #### [RuntimeError: Expected object of backend CUDA but got backend CPU for argument](#error5)  [▶Blog](https://blog.naver.com/jaeyoon_95/221992427221)   
* #### [Failed to initialize NVML: Driver/library version mismatch](#error6)  [▶Blog](https://blog.naver.com/jaeyoon_95/221773869080)   
* #### [TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.](#error7)  [▶Blog](https://blog.naver.com/jaeyoon_95/222109610256)   


---
## error1   
#### error : "RuntimeError: cuda runtime error (710) : device-side assert triggered"   
```
[1,1]<stderr>:RuntimeError: cuda runtime error (710) : device-side assert triggered at /tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCReduceAll.cuh:321
[1,0]<stderr>:/tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = c10::Half, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true]: block: [4,0,0], thread: [96,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
[1,0]<stderr>:/tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = c10::Half, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true]: block: [4,0,0], thread: [97,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
[1,0]<stderr>:/tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = c10::Half, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true]: block: [4,0,0], thread: [98,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
[1,0]<stderr>:/tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = c10::Half, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true]: block: [4,0,0], thread: [99,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
[1,0]<stderr>:/tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = c10::Half, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true]: block: [4,0,0], thread: [100,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
[1,0]<stderr>:/tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = c10::Half, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true][1,0]<stderr>:: block: [4,0,0], thread: [101,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
[1,0]<stderr>:/tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = c10::Half, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true]: block: [4,0,0], thread: [102,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
[1,0]<stderr>:/tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = c10::Half, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true]: block: [4,0,0], thread: [103,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
[1,0]<stderr>:/tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = c10::Half, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true]: block: [4,0,0], thread: [104,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
[1,0]<stderr>:/tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = c10::Half, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true]: block: [4,0,0], thread: [105,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
[1,0]<stderr>:/tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = c10::Half, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true]: block: [4,0,0], thread: [106,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
[1,0]<stderr>:/tmp/pip-req-build-l1dtn3mo/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = c10::Half, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, I
```
#### cause : torch.nn.Embedding function. Setting vocab size(range) < input vocab range. It like 'out of index' error.   
#### solve : Increases vocab size.   
   
---
## error2   
#### error : "RuntimeError: Expected object of scalar type Half but got scalar type Float"   
```
RuntimeError: Expected object of scalar type Half but got scalar type Float for sequence element 1 in sequence argument at position #1 'tensors'
```
#### cause : This is an error caused the data types of the two arrays to be summed are different.   
#### solve : Convert the data type use ".float()" function.   
   
---
## error3   
#### error : "AttributeError: module 'tensorflow' has no attribute 'sub'"   
```
﻿AttributeError: module 'tensorflow' has no attribute 'sub'
```
#### cause : As the Tensorflow version changed, the API changed.     
#### solve : You can use it with the function below.(This is the same for multiplication.)   
```
# before tf.sub
tf.subtract

# before tf.mul
tf.multiply
```   
   
---
## error4   
#### error : "ModuleNotFoundError: No module named 'sklearn'"   
```
﻿ModuleNotFoundError: No module named 'sklearn'
```
#### cause : 'sklearn' library doesn't installed.     
#### solve : You can install it with the command below.   
```
# conda
conda install scikit-learn

# pip
pip3 install -U scikit-learn
```   
   
---
## error5   
#### error : "RuntimeError: Expected object of backend CUDA but got backend CPU for argument"   
```
﻿﻿RuntimeError: Expected object of backend CUDA but got backend CPU for argument
```
#### cause : This error occurs when setting model to gpu(cuda) and loading tensor to cpu.   
#### solve : Set tensor to gpu(cuda).   
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = BertModel.from_pretrained(bert_model)

model.to(device) #model load to GPU

input_tensor = input_tensor.to(device) #tensor load to GPU
```      
   
---
## error6   
#### error : "Failed to initialize NVML: Driver/library version mismatch"   
```
Failed to initialize NVML: Driver/library version mismatch
```
#### cause : This error occurs because nvidia driver kernel module loaded incorrectly.   
#### solve : Unloading and Loading nvidia driver kernel module.   
(1) check loaded nvidia driver kernel.   
```
ailab@ailab:~$ lsmod | grep nvidia
nvidia_drm             45056  5
nvidia_modeset       1093632  8 nvidia_drm
nvidia              18194432  382 nvidia_modeset
drm_kms_helper        172032  1 nvidia_drm
drm                   401408  8 drm_kms_helper,nvidia_drm
ipmi_msghandler        53248  2 ipmi_devintf,nvidia

```
(2) Unloading nvidia driver kernel.   
```
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia_uvm
sudo rmmod nvidia

※If you get "rmmod: ERROR: Module nividia_drm is in use."
Use this command : sudo lsof /dev/nvidia*
```   
(3) Check if the nvidia driver kernel is successfully load.
```
ailab@ailab:~$ nvidia-smi
Sun Oct  4 22:53:35 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:01:00.0  On |                  N/A |
| 95%   91C    P2   138W / 250W |   9287MiB / 11177MiB |     91%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  Off  | 00000000:02:00.0 Off |                  N/A |
| 77%   82C    P2   215W / 250W |   8377MiB / 11178MiB |     91%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0       944      G   ...quest-channel-token=7500496501391111563   168MiB |
|    0      1370      G   /usr/lib/xorg/Xorg                           237MiB |
|    0      2091      G   compiz                                       165MiB |
|    0      6066      C   python                                      8705MiB |
|    0     28597      G   ...b/pycharm-community-2019.2/jbr/bin/java     3MiB |
|    1      6067      C   python                                      8363MiB |
+-----------------------------------------------------------------------------+
```   
   
---
## error7   
#### error : "TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."   
```
TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```
#### cause : This error occurs when allocated GPU tensor convert to type of numpy. X is GPU tensor. if you use X.numpy() function, This error is occur.   
#### solve : Use .cpu() function to GPU tensor.
```
# X is GPU tensor. 
X.numpy() # This causes an error.
X.cpu().numpy() # This does not cause an error.
```      