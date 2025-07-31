from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0  
    for i, j, c in zip(data["user_id"], data["question_id"], data["is_correct"]):
        x = theta[i] - beta[j]
        # using log1p(exp(x)) for stability
        log_lklihood += c * x - np.log1p(np.exp(x))  
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    grad_theta = np.zeros_like(theta)
    grad_beta = np.zeros_like(beta)

    for i, j, c in zip(data["user_id"], data["question_id"], data["is_correct"]):
        x = theta[i] - beta[j]
        p = sigmoid(x)
        # d/d theta_i: (c - p)
        grad_theta[i] += (c - p)
        # d/d beta_j: (-c + p)
        grad_beta[j] += (-c + p)

    theta = theta + lr * grad_theta
    beta = beta + lr * grad_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    # get max between user_id and question_id to ensure we have enough space
    n_users = max(max(data["user_id"]), max(val_data["user_id"])) + 1
    n_questions = max(max(data["question_id"]), max(val_data["question_id"])) + 1
    theta = np.zeros(n_users, dtype=float)
    beta = np.zeros(n_questions, dtype=float)

    val_acc_lst = []
    train_ll, val_ll = [], []

    for i in range(iterations):
        theta, beta = update_theta_beta(data, lr, theta, beta)

        train_ll.append(-neg_log_likelihood(data, theta=theta, beta=beta))
        val_ll.append(-neg_log_likelihood(val_data, theta=theta, beta=beta))
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)

        print("Iter {}/{} \t Train LL: {:.4f} \t Val LL: {:.4f} \t Val Acc: {:.4f}"
        .format(i + 1, iterations, train_ll[-1], val_ll[-1], score))
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        print()

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_ll, val_ll


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

# helper to generate plot of training curves
def plot_irt_ll_curves(train_ll, val_ll, lr, iters):
    iters_axis = np.arange(1, len(train_ll) + 1)
    plt.figure()
    plt.plot(iters_axis, train_ll, label="Train log-likelihood")
    plt.plot(iters_axis, val_ll, label="Validation log-likelihood")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.title(f"IRT training curves (lr={lr}, iters={iters})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    lr_tag = str(lr).replace(".", "p")
    filename = f"irt_lr{lr_tag}_iters{iters}.png"
    plt.savefig(filename)

def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # some candidates for learning rate and iterations
    lr_candidates = [0.005, 0.01, 0.05]
    iter_candidates = [25, 50]

    best = {"val_acc": -1, "lr": None, "iters": None, "theta": None, "beta": None,
            "train_ll_curve": None, "val_ll_curve": None}

    for lr in lr_candidates:
        for iters in iter_candidates:
            print(f"IRT with learning rate={lr}, iterations={iters}")
            theta, beta, val_acc_lst, train_ll_hist, val_ll_hist = irt(
                train_data, val_data, lr=lr, iterations=iters
            )
            plot_irt_ll_curves(train_ll_hist, val_ll_hist, lr, iters)

            if val_acc_lst[-1] > best["val_acc"]:
                best.update({
                    "val_acc": val_acc_lst[-1],
                    "lr": lr,
                    "iters": iters,
                    "theta": theta,
                    "beta": beta,
                    "train_ll_curve": train_ll_hist,
                    "val_ll_curve": val_ll_hist
                })

    # validation and test accuracy with the best hyperparameters
    print("Best hyperparameters")
    print(f"Learning rate = {best['lr']}, iterations = {best['iters']}")
    val_acc = evaluate(val_data, best["theta"], best["beta"])
    print(f"Validation accuracy = {val_acc:.4f}")
    test_acc = evaluate(test_data, best["theta"], best["beta"])
    print(f"Test accuracy = {test_acc:.4f}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    beta = best["beta"]
    theta = best["theta"]

    order = np.argsort(beta)
    j_easy = order[int(0.1 * len(beta))]
    j_mid = order[len(beta) // 2]
    j_hard = order[int(0.9 * len(beta))]
    question_ids = [j_easy, j_mid, j_hard]

    t_min, t_max = float(np.min(theta)), float(np.max(theta))
    t_grid = np.linspace(t_min - 1.0, t_max + 1.0, 200)

    plt.figure()
    for j in question_ids:
        p = sigmoid(t_grid - beta[j])
        plt.plot(t_grid, p, label=f"question j={j} (β={beta[j]:.2f})")

    plt.xlabel(r"Ability $\theta$")
    plt.ylabel(r"$p(c_{ij}=1 \mid \theta, \beta_j)$")
    plt.title("IRT for Three Questions")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("irt_three_questions.png")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
