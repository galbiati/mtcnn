# Pure PyTorch MTCNN

MTCNN end-to-end in PyTorch.

Supports batched inputs by adding a column to the bounding box matrix for batch index.

If you use CUDA, everything stays on device through end of inference.

Python 3 only; 3.7+ recommended.

## Credit
Adapted from [Dan Antoshchenko's implementation](https://github.com/TropComplique/mtcnn-pytorch)
