# Error Collections!

- - -
### Error List
* #### [﻿RuntimeError: cuda runtime error (710) : device-side assert triggered](#error1) [블로그](https://blog.naver.com/jaeyoon_95/222104626159) 
* #### [﻿RuntimeError: Expected object of scalar type Half but got scalar type Float](https://blog.naver.com/jaeyoon_95/222064412708) 
* #### [﻿AttributeError: module 'tensorflow' has no attribute 'sub'](https://blog.naver.com/jaeyoon_95/222007030881) 


---
## #error1   
### error : RuntimeError: cuda runtime error (710) : device-side assert triggered   
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
### cause : torch.nn.Embedding function. Setting vocab size(range) < input vocab range. It like 'out of index' error.   
### solve : Increases vocab size.   