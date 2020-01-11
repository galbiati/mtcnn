# Pure PyTorch MTCNN

MTCNN end-to-end in PyTorch.

Supports batched inputs by adding a column to the bounding box matrix for batch index.

Keeps everything on CUDA device until done, if you like.

## Credit
Adapted from [Dan Antoshchenko's implementation](https://github.com/TropComplique/mtcnn-pytorch)
