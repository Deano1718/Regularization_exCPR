# Minimizing Chebyshev Prototype Risk Magically Mitigates the Perils of Overfitting

The exCPR algorithm regularizes deep neural networks during training based on the simple notion that minimizing the variation of angle between feature vectors and their prototypes coupled with maximizing the angular distance between class prototypes on the training set can mitigate the risk of overfitting to the training examples.  For the precise form and derivation of why this is the case, please see our paper referenced below.  A novel outcome of the algorithm is that is computes an approximate covariance matrix loss contribution in O(J lg J), where J is the number of features.

## Simple Implementation

The implementation has been designed to fit a wide variety of use cases in a similar manner to existing PyTorch design patterns.  exCPR requires instantiation of two objects: Prototype (to update and maintain the class prototypes during training) and the exCPRLoss loss module that contains all the loss computations for exCPR.  The additional modification exCPR requires is the output of the feature vector (at a layer selected by the user) when the current batch of data points is forwarded.  Our code provides one simple way this can be achieved.

```python
#Create Prototype object and pass object reference to exCPRLoss module

prototypes = Prototype(nclass, num_ftrs)
criterion = exCPRLoss(prototypes, verbose=args.verbose)

#training loop
for epoch in range(1,num_epochs+1):
      for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            ftr_vec, outputs = cur_model(inputs)
            loss = criterion(ftr_vec, outputs, labels)

            loss.backward()

            prototypes.step()
            optimizer.step()                       

```
## On Hyperparameters
exCPR can at first seem daunting due to the number of hyperparameters on the loss components (and r).  In lieu of a large hyperparameter study, we recommend r=10 to start and then set the exCPR loss component weights to 1.0 to begin with.  The goal should be that all exCPR loss components should decrease significantly from their average values in the first 10 epochs.  If by the end of a training a particular loss component has not decreased, increase its loss weight by an order of magnitude.  Repeat this process until all loss components decrease steadily on the training set.

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
