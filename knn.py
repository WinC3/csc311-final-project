import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]

    # use user-based KNN imputation
    val_accuracies_user = []
    for k in k_values:
        accuracy = knn_impute_by_user(sparse_matrix, val_data, k)
        val_accuracies_user.append(accuracy)

    # report accuracy on validation data 
    print("Validation accuracies for k values:")
    for k, accuracy in zip(k_values, val_accuracies_user):
        print("k = {}, accuracy = {}".format(k, accuracy))

    # plot
    plt.figure()
    plt.plot(k_values, val_accuracies_user, marker="o")
    plt.xticks(k_values)
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.title("User-based KNN: Validation Accuracy vs k")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("knn_user_val_accuracy.png") 
    plt.show()

    # compute test accuracy with the best k
    best_k_user = k_values[np.argmax(val_accuracies_user)]
    test_accuracy_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    print("Test Accuracy with k = {}: {}".format(best_k_user, test_accuracy_user))

    # use item-based KNN imputation
    val_accuracies_item = []
    for k in k_values:
        accuracy = knn_impute_by_item(sparse_matrix, val_data, k)
        val_accuracies_item.append(accuracy)

    # report accuracy on validation data
    print("Validation accuracies for item-based KNN:")
    for k, accuracy in zip(k_values, val_accuracies_item):
        print("k = {}, accuracy = {}".format(k, accuracy))
    
    # plot
    plt.figure()
    plt.plot(k_values, val_accuracies_item, marker="o")
    plt.xticks(k_values)
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.title("Item-based KNN: Validation Accuracy vs k")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("knn_item_val_accuracy.png")
    plt.show()

    # compute test accuracy with the best k
    best_k_item = k_values[np.argmax(val_accuracies_item)]
    test_accuracy_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)
    print("Test Accuracy with k = {}: {}".format(best_k_item, test_accuracy_item))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
