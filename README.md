# Minimizing Chebyshev Prototype Risk Magically Mitigates the Perils of Overfitting

The exCPR algorithm regularizes deep neural networks during training based on the simple notion that minimizing the variation of angle between feature vectors and their prototypes coupled with maximizing the angular distance between class prototypes on the training set can mitigate the risk of overfitting to the training examples.  For the precise form and derivation of why this is the case, please see our paper referenced below.

## Simple Implementation

```python
import torch as ch
from torchvision import transforms

train_transform = transforms.Compose([...])

data_path = "robust_CIFAR"

train_data = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_ims")))
train_labels = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_lab")))
train_set = folder.TensorDataset(train_data, train_labels, transform=train_transform) 
```

## Citation 
```
@misc{dean2024minimizing,
      title={Minimizing Chebyshev Prototype Risk Magically Mitigates the Perils of Overfitting}, 
      author={Nathaniel Dean and Dilip Sarkar},
      year={2024},
      eprint={2404.07083},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
