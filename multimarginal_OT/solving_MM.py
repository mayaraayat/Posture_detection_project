import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pickle


def solve_multimarginal_optimal_transport(cost_matrix):
    """
    Solves the multimarginal optimal transport problem based on the given cost matrix.

    Args:
        cost_matrix (ndarray): The cost matrix representing the transportation costs between clusters.

    Returns:
        list: A list of tuples representing the optimal mapping between source and target clusters.
              Each tuple contains the index of the source cluster and the index of the target cluster.
    """
    print(cost_matrix, len(cost_matrix))
    num_clusters = len(cost_matrix)

    # Create a new model
    # Create a new model
    model = gp.Model("MultimarginalOptimalTransport")

    # Create decision variables
    x = model.addVars(num_clusters, num_clusters, vtype=GRB.BINARY, name="x")

    # Set objective function
    obj = sum(
        sum(
            sum(
                sum(
                    cost_matrix[i][j][comp][k][l] * x[i, j]
                    for l in range(3)
                    
                )
                for k in range(len(cost_matrix[i][j][comp]))
            )
            for comp in range(2)
        )
        for i in range(num_clusters)
        for j in range(len(cost_matrix[0]))
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Add constraints
    for i in range(num_clusters):
        model.addConstr(sum(x[i, j] for j in range(
           len(cost_matrix[0])) ) <= 1, name=f"source_cluster_{i + 1}_constraint")

    for j in  range(len(cost_matrix[0])):#range(min(num_clusters,len(cost_matrix[0]))):
        model.addConstr(sum(x[i, j] for i in range(num_clusters) )
                        == 1, name=f"target_cluster_{j + 1}_constraint")

    # Set solver parameters (optional)
    model.setParam("OutputFlag", 0)  # Disable solver output

    # Set the maximum number of iterations
    max_iterations = 1000

    objective_history = []  # Initialize objective_history as an empty list

    def capture_objective_value(model, where):
        if where == GRB.Callback.MIPSOL:
            objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            objective_history.append(objective_value)

    # Optimize the model

    for iteration in range(max_iterations):
        model.optimize(capture_objective_value)

    # Print the objective function value history
    #print("Objective Function Value History:")
    # print(objective_history)

    # Plot the objective function value history
    # plot_objective_history(objective_history)
    # Check optimization status

    if model.status == GRB.OPTIMAL:
        # Retrieve optimal solution
        solution = []
        for i in range(num_clusters):
            for j in range(num_clusters):
                if x[i, j].x > 0.5:  # Consider a mapping if the variable value is close to 1
                    solution.append((i, j))

        return solution
    else:
        print("No feasible solution found.")
        return None


def plot_objective_history(objective_history):
    """
    Plots the objective function value history.

    Args:
        objective_history (list): List of objective function values at each iteration.

    Returns:
        None
    """
    plt.plot(range(len(objective_history)), objective_history)
    plt.title('Objective Function Value History')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.show()


def visualize_optimal_mapping(solution, cost_matrix):
    """
    Visualizes the optimal mapping between source and target clusters.

    Args:
        solution (list): List of tuples representing the optimal mapping.
        cost_matrix (ndarray): The cost matrix.

    Returns:
        None
    """
    if solution:
        source_clusters, target_clusters = zip(*solution)
        plt.figure(figsize=(6, 6))
        plt.scatter(source_clusters, target_clusters, c='b')
        plt.title('Optimal Mapping')
        plt.xlabel('Source Cluster')
        plt.ylabel('Target Cluster')
        plt.xticks(range(len(cost_matrix)))
        plt.yticks(range(len(cost_matrix)))
        plt.grid(True)
        plt.show()
    else:
        print("No feasible solution found.")
