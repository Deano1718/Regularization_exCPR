# Minimizing Chebyshev Prototype Risk Magically Mitigates the Perils of Overfitting

The exCPR algorithm regularizes deep neural networks during training based on the simple notion that minimizing the variation of angle between feature vectors and their prototypes coupled with maximizing the angular distance between class prototypes on the training set can mitigate the risk of overfitting to the training examples.  For the precise form and derivation of why this is the case, please see our paper referenced below.

## Simple Implementation

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
