# Link_Predcition_Methods

Rewrite [link-prediction](https://github.com/lucashu1/link-prediction) with Python3 in Windows 10.

## Requirements

- python 3.6
- [networkx](https://github.com/networkx/networkx)
- tensorflow (1.0 or later)
- gensim
- scikit-learn

## Included Files

### Network Data
* `facebook/facebook/`: Original [Facebook ego networks](https://snap.stanford.edu/data/egonets-Facebook.html) dataset, with added `.allfeat` files (with both ego and alter features)
* `facebook/fb-processed/`: Pickle dumps of (adjacency_matrix, feature_matrix) tuples for each ego network
* `twitter/twitter`: Original [Twitter ego networks](https://snap.stanford.edu/data/egonets-Twitter.html) dataset
* `train-test-splits/`: Pickle dumps of pre-processed train-test splits for Facebook ego networks, with varying degrees of visibility (i.e. how many edges are hidden). Includes: `adj_train`, `train_edges`, `train_edges_false`, `val_edges`, `val_edges_false`, `test_edges`, `test_edges_false`
* `process-fb-networks.py`: Script used to process raw Facebook data and generate pickle dumps
* `process-combined-network.py`: Script used to combine Facebook ego networks and generate complete network pickle dump
* `fb-train-test-splits.py`: Script used to generate and store train-test splits for each Facebook ego network


## Run Step

### Facebook Dataset

- Step1: Unzip Facebook Dataset in `facebook/`, Run `facebook/allfeat_generate.py` to generate `.allfeat`

- Step2: Run `facebook/process-fb-networks.py` to save `(adj, features)` in pickle file

- Step3: Run `facebook/process-combined-networks.py` to combine all ego networks (including node features) and store in (adj, features) tuple

> But it seems that it makes feature value 0 becomes 1 and 1 becomes 2.

- Step4: Run `facebook/fb-train-test-splits.py` to generate Train-Test splits.