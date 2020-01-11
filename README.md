# Pure PyTorch MTCNN

MTCNN end-to-end in PyTorch, as a single module.

Supports batched inputs by adding a column to the bounding box matrix for batch index.

If you use CUDA, everything stays on device through end of inference.

Does NOT include file read/pipelining - implement this how you want. 

Pretrained weights are included in `mtcnn.pth`.

### Preprocessing requirements
- Images must be dtype `float32`
- To normalize from `uint8` image, do `(image - 127.5) / 128`
- Inputs must be batched and in `(N, C, H, W)` format
- Channel order should be RGB (watch out OpenCV users)

### Model output structure
Each item in a batch may have a variable number of output bounding boxes, so a tensor maintaining a batch axis cannot be used. Instead, the model `MTCNN` will output a *flattened matrix* with the following stucture:

> Size: [num_boxes, 20]
>
> Each row is a single bounding box.
>
> Column 0 is batch index.
> 
> Columns 1 - 4 are bounding box top left and bottom right coordinates.
>
> Column 5 is score for that box.
>
> Columns 6-10 are offset values
>
> Columns 10-20 are landmark coordinates (same order as output by `ONet`)


## Example
```python
import cv2
import numpy as np
import torch

from mtcnn import MTCNN

# Config
path_to_saved_model = '...' # Fill in your own paths here
path_to_test_image = '...'

device = torch.device('...')    # Use 'cuda' for cuda, 'cpu' for cpu

# Load model
mtcnn = MTCNN()

state_dict = torch.load(path_to_saved_model)
mtcnn.load(state_dict)
mtcnn.eval()
mtcnn.to(device)

# Placeholder data loading pipeline
image = cv2.imread(path_to_test_image, cv2.IMREAD_COLOR)    # unit8, BGR, HWC
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)              # Convert to RGB
image = (image.astype(np.float32) - 127.5) / 128            # Convert to float32 and normalize
image = image.transpose(2, 0, 1)                            # Switch to CHW

# Send to device as torch tensor, add batch axis, and run inference
image = torch.as_tensor(image, dtype=torch.float32, device=device).unsqueeze(0)
bounding_boxes = mtcnn(image)
```


## Requirements
- `numpy>=1.17.1`
- `torch>=1.3.1`
- `torchvision>=0.4.2`

Python 3 only; 3.7+ recommended.


## Credit
Adapted from [Dan Antoshchenko's implementation](https://github.com/TropComplique/mtcnn-pytorch).
