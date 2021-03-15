# Link_Predcition_Methods

Rewrite [link-prediction](https://github.com/lucashu1/link-prediction) with Python3 in Windows 10.

## Requirements

- python 3.6
- [networkx]([networkx/networkx: Network Analysis in Python (github.com)](https://github.com/networkx/networkx))

## Included Files

### Network Data
* `facebook/facebook/`: Original [Facebook ego networks](https://snap.stanford.edu/data/egonets-Facebook.html) dataset, with added `.allfeat` files (with both ego and alter features)
* `facebook/fb-processed/`: Pickle dumps of (adjacency_matrix, feature_matrix) tuples for each ego network
* `twitter/twitter`: Original [Twitter ego networks](https://snap.stanford.edu/data/egonets-Twitter.html) dataset