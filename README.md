# Naive-Bayes-and-Tree-Augmented-Naive-Bayes
Implement the naive Bayes and TAN (tree-augmented naive Bayes) for binary classification problems.

1. Assume all of the variables are discrete valued.

2. Use Laplace estimates (pseudocounts of 1) to estimate all probabilities.

3. Use Prim's algorithm to find a maximal spanning tree. If there are ties in selecting maximum weight edges, use the following preference criteria: (1) prefer edges emanating from variables listed earlier in the input file, (2) if there are multiple maximum weight edges emanating from the first such variable, prefer edges going to variables listed earlier in the input file.

4. Use the first variable in the input file as the root.

5. Run the script "bayes" by: 

   bayes lymph_train.arff lymph_test.arff n

   bayes lymph_train.arff lymph_test.arff t

   bayes vote_train.arff vote_test.arff n

   bayes vote_train.arff vote_test.arff t
