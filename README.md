# Error Collections!

- - -
### Error List
* #### [﻿RuntimeError: cuda runtime error (710) : device-side assert triggered](#error1)  [▶Blog](https://blog.naver.com/jaeyoon_95/222104626159) 
* #### [﻿RuntimeError: Expected object of scalar type Half but got scalar type Float](#error2)  [▶Blog](https://blog.naver.com/jaeyoon_95/222064412708) 
* #### [﻿AttributeError: module 'tensorflow' has no attribute 'sub'](#error3)  [▶Blog](https://blog.naver.com/jaeyoon_95/222007030881)   
* #### [ModuleNotFoundError: No module named 'sklearn'](#error4)  [▶Blog](https://blog.naver.com/jaeyoon_95/222007026711)   
* #### [RuntimeError: Expected object of backend CUDA but got backend CPU for argument](#error5)  [▶Blog](https://blog.naver.com/jaeyoon_95/221992427221)   


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