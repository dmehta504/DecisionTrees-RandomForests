import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """
    # Decision Tree
    #              A1 == 0
    #              /    \
    #             /      \
    #         A4 == 0     1
    #           /  \
    #          /    \
    #         /      \
    #     A3 == 0  A2 == 0
    #       / \      /  \
    #      1   0     1   0

    decision_tree_root = DecisionNode(None, None, lambda a1: a1[0] == 0)
    node_a2 = DecisionNode(None, None, lambda a2: a2[1] == 0)
    node_a3 = DecisionNode(None, None, lambda a3: a3[2] == 0)
    node_a4 = DecisionNode(None, None, lambda a4: a4[3] == 0)

    # Build Root
    decision_tree_root.left = node_a4
    decision_tree_root.right = DecisionNode(None, None, None, 1)

    # Build Left (A4)
    node_a4.left = node_a3
    node_a4.right = node_a2

    # Build Left of A4 (A3)
    node_a3.left = DecisionNode(None, None, None, 1)
    node_a3.right = DecisionNode(None, None, None, 0)

    # Build Right of A4 (A2)
    node_a2.left = DecisionNode(None, None, None, 1)
    node_a2.right = DecisionNode(None, None, None, 0)

    # TODO: finish this.
    # raise NotImplemented()

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    # TODO: finish this.
    # Creates a 2x2 array
    # [0 , 0
    #  0,  0]
    matrix = np.array([[0, 0], [0, 0]])

    for i in range(len(classifier_output)):
        # If we classify correctly check for True Positive or True Negative
        if classifier_output[i] == true_labels[i]:
            if true_labels[i] == 1:
                matrix[0, 0] = matrix[0, 0] + 1  # True Positive
            else:
                matrix[1, 1] = matrix[1, 1] + 1  # True Negative

        # If we classify incorrectly check for False Positive or False Negative
        elif classifier_output[i] != true_labels[i]:
            if true_labels[i] == 0:
                matrix[1, 0] = matrix[1, 0] + 1  # False Positive
            else:
                matrix[0, 1] = matrix[0, 1] + 1  # False Negative

    return matrix
    # raise NotImplemented()


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """

    # TODO: finish this.
    matrix = confusion_matrix(classifier_output, true_labels)
    true_positive = matrix[0, 0]
    false_positive = matrix[1, 0]

    return true_positive / (true_positive + false_positive)
    # raise NotImplemented()


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """

    # TODO: finish this.
    matrix = confusion_matrix(classifier_output, true_labels)
    true_positive = matrix[0, 0]
    false_negative = matrix[0, 1]

    return true_positive / (true_positive + false_negative)
    # raise NotImplemented()


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """
    matrix = confusion_matrix(classifier_output, true_labels)
    true_positive = matrix[0, 0]
    true_negative = matrix[1, 1]

    return (true_positive + true_negative)/len(true_labels)
    # TODO: finish this.
    # raise NotImplemented()


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    number_unique_class, number_counts = np.unique(class_vector, return_counts=True)
    gini_impurity_for_class = []

    for i in range(len(number_unique_class)):
        # Calculate gini impurity for each class as p_i * (1 - p_i)
        p_i = number_counts[i] / len(class_vector)
        gini_impurity_for_class.append(p_i * (1.0 - p_i))

    g_impurity = np.sum(gini_impurity_for_class)  # Total gini impurity is sum of gini impurity for each class
    return g_impurity
    # raise NotImplemented()


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    g_impurity_previous = gini_impurity(previous_classes)

    # Calculate the total size of future classes
    total = 0.0
    for i in range(len(current_classes)):
        total = total + len(current_classes[i])

    if total == 0:
        return 0

    # Calculate gini impurity of each future class (split)
    gain = 0.0
    for i in range(len(current_classes)):
        g_impurity = gini_impurity(current_classes[i])
        gain += (len(current_classes[i])/total) * g_impurity

    return g_impurity_previous - gain
    # raise NotImplemented()


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        # Check for termination

        # No more samples left i.e. no more X values remaining
        if features.shape[0] <= 1:
            return DecisionNode(None, None, None, int(np.mean(classes)))

        # If all classes are the same, then return the class
        if np.unique(classes).shape[0] == 1:
            return DecisionNode(None, None, None, classes[0])

        # If max depth reached, then return most frequent class
        if depth == self.depth_limit:
            class_values, class_count = np.unique(classes, return_counts=True)
            most_frequent_index = np.argmax(class_count)
            return DecisionNode(None, None, None, class_values[most_frequent_index])

        else:
            # Find the feature with the highest normalized Gini Gain
            best_index = self.select_splitval(features, classes)
            # Choose the split value as the mean of the best feature found to split on
            alpha_best = np.median(features[:, best_index])
            max_feature = np.max(features[:, best_index])
            if max_feature == alpha_best:
                return DecisionNode(None, None, None, int(np.mean(classes)))

            # Recursively build the left and right subtree
            root = DecisionNode(None, None, lambda feature: feature[best_index] <= alpha_best)
            left_index = np.where(features[:, best_index] <= alpha_best)  # Get the indices for left tree
            right_index = np.where(features[:, best_index] > alpha_best)  # Get the indices for right tree
            root.left = self.__build_tree__(features[left_index], classes[left_index], depth + 1)
            root.right = self.__build_tree__(features[right_index], classes[right_index], depth + 1)
            return root
        # raise NotImplemented()

    def select_splitval(self, features, classes):
        gain = []
        previous_classes = classes.tolist()

        # For each feature in features, calculate the Gini Gain obtained by splitting on that feature
        for i in range(features.shape[1]):
            feature = features[:, i]
            # Split on the median value for that particular feature, this ensures we split the data in half
            split = np.median(feature)
            left_split = np.where(feature[feature[:] <= split])
            right_split = np.where(feature[feature[:] > split])
            current_classes = [classes[left_split], classes[right_split]]
            # Calculate the gini gain obtained by splitting on the selected feature
            gain.append(gini_gain(previous_classes, current_classes))

        best_index = np.argmax(gain)  # Finds the index of the feature that has the highest Gini Gain
        return best_index

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """

        class_labels = [self.root.decide(feature) for feature in features]

        # TODO: finish this.
        # raise NotImplemented()
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """

    # TODO: finish this.
    features = dataset[0]
    classes = dataset[1]
    num_of_bags = int(len(np.column_stack((features, classes))) / k)  # n // k i.e. number of samples in each bag
    data = np.column_stack((features, classes))
    folds = []
    for i in range(k):
        copy = np.copy(data)
        np.random.shuffle(copy)  # Shuffle the data
        # Create an index array
        index_array = np.arange(0, features.shape[0], dtype=int)
        # Create a slice to make a testing set with n // k data points
        index_slice = np.random.choice(index_array, num_of_bags, replace=False)
        test_data = copy[index_slice, :]
        train_data = np.delete(copy, index_slice, 0)  # Delete the points used in testing set to get the training set
        # Get the features
        train_features = train_data[:, 0:-1]
        test_features = test_data[:, 0:-1]
        # Get the classes
        train_classes = train_data[:, -1]
        test_classes = test_data[:, -1]
        # Add the fold i.e. (training set, testing set)
        folds.append(((train_features, train_classes), (test_features, test_classes)))

    return folds
    # raise NotImplemented()


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.attributes_used = []  # Creating a list to track which attributes were used to train a specific tree

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        # Converting features and classes into numpy for slicing/indexing operations
        features = np.asarray(features)
        classes = np.asarray(classes)

        for i in range(self.num_trees):
            # Create an index array, then at random (w/ replacement) choose samples to create a training set
            # Size of sample is based on example_subsample_rate
            index_array = np.arange(0, features.shape[0], dtype=int)
            sample_slice = np.random.choice(index_array, size=int(self.example_subsample_rate * features.shape[0]),
                                            replace=True)
            # Get the training features and classes for our subsample
            train_classes = classes[sample_slice]
            train_features = features[sample_slice]

            # From above sample, choose attributes at random to learn on, size is based on attr_subsample_rate
            attribute_slice = np.random.randint(features.shape[1], size=int(self.attr_subsample_rate * features.shape[1]))
            train_features = train_features[:, attribute_slice]

            tree = DecisionTree(self.depth_limit)
            tree.fit(train_features, train_classes)
            self.trees.append(tree)
            self.attributes_used.append(attribute_slice)

        # raise NotImplemented()

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """

        # TODO: finish this.
        votes = []
        features = np.asarray(features)
        # For each tree in the forest, get the classifications from each tree
        # based on the features used to build tree
        for i in range(self.num_trees):
            tree = self.trees[i]
            features_used = features[:, self.attributes_used[i]]
            votes.append(tree.classify(features_used))

        votes = np.array(votes)
        classifications = []
        # Based on the votes from each tree, return the class that most frequently appears
        for i in range(len(features)):
            classes = votes[:, i]
            class_val, class_count = np.unique(classes, return_counts=True)
            # Get the classification that appears most frequently and add it to the list
            classifications.append(class_val[np.argmax(class_count)])

        return classifications
        # raise NotImplemented()


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        """Create challenge classifier.
        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        self.trees = []
        self.depth_limit = 4
        self.num_trees = 10
        self.example_subsample_rate = 0.4
        self.attr_subsample_rate = 0.9
        self.attributes_used = []  # Creating a list to track which attributes were used to train a specific tree
        # raise NotImplemented()

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        # Converting features and classes into numpy for slicing/indexing operations
        features = np.asarray(features)
        classes = np.asarray(classes)

        for i in range(self.num_trees):
            # Create an index array, then at random (w/ replacement) choose samples to create a training set
            # Size of sample is based on example_subsample_rate
            index_array = np.arange(0, features.shape[0], dtype=int)
            sample_slice = np.random.choice(index_array, size=int(self.example_subsample_rate * features.shape[0]),
                                            replace=True)
            # Get the training features and classes for our subsample
            train_classes = classes[sample_slice]
            train_features = features[sample_slice]

            # From above sample, choose attributes at random to learn on, size is based on attr_subsample_rate
            attribute_slice = np.random.choice(range(0, features.shape[1]),
                                               size=int(self.attr_subsample_rate * features.shape[1]), replace=False)

            attribute_slice = np.sort(attribute_slice)
            train_features = train_features[:, attribute_slice]

            tree = DecisionTree(self.depth_limit)
            tree.fit(train_features, train_classes)
            self.trees.append(tree)
            self.attributes_used.append(attribute_slice)

        # raise NotImplemented()

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """

        # TODO: finish this.
        votes = []
        features = np.asarray(features)
        # For each tree in the forest, get the classifications from each tree
        # based on the features used to build tree
        for i in range(self.num_trees):
            tree = self.trees[i]
            features_used = features[:, self.attributes_used[i]]
            votes.append(tree.classify(features_used))

        votes = np.array(votes)
        classifications = []
        # Based on the votes from each tree, return the class that most frequently appears
        for i in range(len(features)):
            classes = votes[:, i]
            class_val, class_count = np.unique(classes, return_counts=True)
            # Get the classification that appears most frequently and add it to the list
            classifications.append(class_val[np.argmax(class_count)])

        return classifications
        # raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        data_np = np.array(data)
        data = (data_np * data_np) + data_np
        return data
        # raise NotImplemented()

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.
        data_np = np.array(data)
        data_np_slice = data_np[0:100, :]
        data_np_slice = np.sum(data_np_slice, axis=1)  # Sum along all each row in the slice

        return (data_np_slice[np.argmax(data_np_slice)], np.argmax(data_np_slice))
        # raise NotImplemented()

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        # TODO: finish this.
        data_np_flat = np.array(data).flatten()
        data_positive = data_np_flat[data_np_flat > 0.0]
        values, number_of_occurrences = np.unique(data_positive, return_counts=True)
        list_of_occurrences = []
        for i in range(len(values)):
            list_of_occurrences.append((values[i], number_of_occurrences[i]))

        return list_of_occurrences
        # raise NotImplemented()


def return_your_name():
    # return your name
    # TODO: finish this
    return "Dhruv Mehta"
    # raise NotImplemented()
