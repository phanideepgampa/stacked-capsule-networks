# stacked-capsule-networks
Pytorch Implementation of the paper **StackedCapsuleAutoEncoders**. [link](https://arxiv.org/abs/1906.06818)
- The code currently supports MNIST dataset only
- All the operations are vectorised for supporting batch operations except the **template transformations** part (Pytorch doesn't have batch support for image transformation functions)
- The code for the set transformers part (setmodules.py) is used from the official repository [https://github.com/juho-lee/set_transformer]. 
- Feel free to cite this repository if you want to use the code.
