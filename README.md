# stacked-capsule-networks
Pytorch Implementation of the paper **StackedCapsuleAutoEncoders**. [link](https://arxiv.org/abs/1906.06818)
- The code currently supports MNIST dataset only
- The code for the set transformers part (setmodules.py) is used from the official repository [https://github.com/juho-lee/set_transformer]. 
- Feel free to cite this repository if you want to use the code.
- To run the code with default parameters according to the paper on MNIST dataset use `python main.py`


## Requirements
- Python 3.8+, PyTorch >= 1.10, torchvision, tqdm, numpy

## Notes on the implementation
The model follows the [official TensorFlow implementation](https://github.com/google-research/google-research/tree/master/stacked_capsule_autoencoders) of the paper:
- Inputs are raw MNIST pixels in [0, 1]. Templates are also kept in [0, 1] (`relu1`), and the image is modelled as a mixture of the transformed templates plus a learned background. The mixing logits are template intensity divided by a learned temperature, plus the log presence of each part.
- Part poses are mapped to affine matrices with `geometric_transform`. Templates are warped with the *inverse* pose, so the pose maps template coordinates into image coordinates.
- Object capsules predict an object-viewer matrix (OV), dynamic object-part corrections that are added to static learned OP matrices, presences and per-vote scales (softplus, bounded away from 0). All likelihoods are computed in log space with `logsumexp`.
- Sparsity losses: an L2 prior on capsule presences, and an entropy posterior that is minimised within an example and maximised between examples.
- Evaluation: each object capsule gets the label that occurs most often among training images where it is the most active capsule. Test images are classified with the label of their most active capsule.
