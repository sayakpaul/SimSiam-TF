# SimSiam-TF

Minimal implementation of SimSiam ((Exploring Simple Siamese Representation Learning)[https://arxiv.org/abs/2011.10566)] by Xinlei Chen & Kaiming He) in TensorFlow 2.

The purpose of this repository is to demonstrate the workflow of SimSiam and NOT to implement it note to note and at the same time I will try not to miss out on the major bits discussed in the paper. For that matter, I'll be using the **Flowers dataset**. 

Following depicts the workflow of SimSiam (taken from the paper) - 

<center>
<img src="https://i.ibb.co/37pNQTP/image.png" width=550></img>
</center>

The authors have also provided a PyTorch-like psuedocode in the paper (how cool!) - 

```python
# f: backbone + projection mlp
# h: prediction mlp
for x in loader: # load a minibatch x with n samples
    x1, x2 = aug(x), aug(x) # random augmentation
    z1, z2 = f(x1), f(x2) # projections, n-by-d
    p1, p2 = h(z1), h(z2) # predictions, n-by-d
    L = D(p1, z2)/2 + D(p2, z1)/2 # loss
    L.backward() # back-propagate
    update(f, h) # SGD update

def D(p, z): # negative cosine similarity
    z = z.detach() # stop gradient
    p = normalize(p, dim=1) # l2-normalize
    z = normalize(z, dim=1) # l2-normalize
    return -(p*z).sum(dim=1).mean()
```

The authors emphasize the `stop_gradient` operation that helps the network to avoid collapsing solutions. Further details about this are available in the paper. SimSiam eliminates the need for using large batch sizes, momentum encoders, memory banks, negative samples, etc. that are important components of the modern self-supervised learning frameworks for visual recognition. This makes SimSiam an easily approachable framework for practical problems.  

## About the notebooks

* `SimSiam_Pre_training.ipynb`: Pre-trains a ResNet50 using **SimSiam**. 
* `SimSiam_Evaluation.ipynb`: Evaluates (linear evaluation) ResNet50 as pre-trained in `SimSiam_Pre_training.ipynb`. 

## Results

| Pre-training Schedule 	| Validation Accuracy  (Linear Evaluation) 	|
|:---------------------:	|:----------------------------------------:	|
|       50 epochs       	|                  45.64%                  	|
|       75 epochs       	|                  44.91%                  	|

**Supervised training** (results are taken from [here](https://github.com/ayulockin/SwAV-TF/blob/master/linear_evaluation/Linear_Evaluation_Fully_Supervised.ipynb) and [here](https://github.com/ayulockin/SwAV-TF/blob/master/Fully_Supervised_from_Scratch.ipynb)):

|                 Training Type                 	| Validation Accuracy  (Linear Evaluation) 	|
|:---------------------------------------------:	|:----------------------------------------:	|
| Supervised ImageNet-trained ResNet50 Features 	|                  48.36%                  	|
|      From Scratch Training with ResNet50      	|                  63.64%                  	|


## Observations

The figure below shows the training loss plots from two different pre-training schedules (50 epochs and 75 epochs) - 

![](https://i.ibb.co/TtXw2Zj/image.png)

We see that the loss gets plateaued after 35 epochs. We can experiment with the following components to further improve this - 

* data augmentation pipeline
* architectures of the two MLP heads
* learning schedule used during pre-training

and so on.  

## Pre-trained weights

* 50 epochs
    * [Projection](https://github.com/sayakpaul/SimSiam-TF/releases/download/v1.0.0/projection.h5)
    * [Prediction](https://github.com/sayakpaul/SimSiam-TF/releases/download/v1.0.0/prediction.h5)
* 75 epochs
    * [Projection](https://github.com/sayakpaul/SimSiam-TF/releases/download/v1.0.0/projection_75.h5)
    * [Prediction](https://github.com/sayakpaul/SimSiam-TF/releases/download/v1.0.0/prediction_75.h5)
    
## Acknowledgements

Thanks to [Connor Shorten's video](https://www.youtube.com/watch?v=k-PcMBYQsOY) on the paper that helped in understanding the paper briefly. Thanks to the [ML-GDE program](https://developers.google.com/programs/experts/) for providing GCP Credits that helped in preparing the experiments. 
