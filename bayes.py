from functions import *

data_train, meta_train, ordered_meta_train = get_data_meta(str(sys.argv[1]))
data_test, meta_test, ordered_meta_test = get_data_meta(str(sys.argv[2]))
method = str(sys.argv[3])

if method == 'n':
    bayes_graph = get_bayes_graph(meta_train, ordered_meta_train)  # Build the graph
    param = learning(bayes_graph, data_train, meta_train)  # Learn the parameters
    correct_num = inference(data_test, meta_test, param)  # Inference on the test set
elif method == 't':
    tan_graph = get_tan_graph(data_train, meta_train, ordered_meta_train)  # Build the graph
    param = learning(tan_graph, data_train, meta_train)  # Learn the parameters
    correct_num = inference(data_test, meta_test, param)  # Inference on the test set

