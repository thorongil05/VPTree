# VPTree

This is a python implementation of the Vantage Point Tree index.

## Definition
The Vantage Point Tree is an index based on exact similarity search. The aim is to search an object faster than the sequential scanning.
The structure is a static balanced binary tree, so for definition:
- it does not allow insertion
- it does not allow deletion
- the nodes are always full
- a node has at most two children

The most important operations that a Vantage Point Tree must have are:
1. tree creation
2. search objects starting from a given query

### Creation
The Vantage Point Tree uses the ball partition: it recursively divides given data sets X.
```
choose vantage point p of X
S1 = {x in X – {p} | d(x,p) ≤ m}
S2 = {x in X – {p} | d(x,p) ≥ m}
```
The equality sign in the formulas ensures the balancing.

### Searching
//TODO

## Implementation
//TODO

## Performance Evaluation
//TODO
