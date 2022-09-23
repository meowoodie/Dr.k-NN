Distributionally Robust Weighted k-Nearest Neighbors
===

Learning a robust classifier from a few samples remains a key challenge in machine learning. A major thrust of research has been focused on developing k-nearest neighbor (k-NN) based algorithms combined with metric learning that captures similarities between samples. When the samples are limited, robustness is especially crucial to ensure the generalization capability of the classifier. In this paper, we study a minimax distributionally robust formulation of weighted k-nearest neighbors, which aims to find the optimal weighted k-NN classifiers that hedge against feature uncertainties. We develop an algorithm, Dr.k-NN, that efficiently solves this functional optimization problem and features in assigning minimax optimal weights to training samples when performing classification. These weights are class-dependent, and are determined by the similarities of sample features under the least favorable scenarios. When the size of the uncertainty set is properly tuned, the robust classifier has a smaller Lipschitz norm than the vanilla k-NN, and thus improves the generalization capability. We also couple our framework with neural-network-based feature embedding. We demonstrate the competitive performance of our algorithm compared to the state-of-the-art in the few-training-sample setting with various real-data experiments.

![architecture](https://github.com/meowoodie/Dr.k-NN/blob/master/illustration.png)
> An illustrative comparison of Dr.k-NN and vanilla k-NN. 

![architecture](https://github.com/meowoodie/Dr.k-NN/blob/master/architecture.png)
> An overview of the end-to-end learning framework. 

### Reference
- [Shixiang Zhu, Liyan Xie, Minghe Zhang, Rui Gao, Yao Xie. *Distributionally Robust Weighted k-Nearest Neighbors*, Annual Conference on Neural Information Processing Systems (NeurIPS), 2022.](https://arxiv.org/abs/2006.04004)
