---
title: "Project Spotlight: Mastering the Machine Learning Toolkit"
excerpt: "A deep dive into supervised, unsupervised, randomized optimization, and reinforcement learning algorithms using Scikit-learn, Matplotlib, Gymnasium, and custom libraries."
collection: portfolio
header:
  teaser: "images/ml-toolkit-exploration/learning_curves.png" # Using one of the allowed images
permalink: /portfolio/ml-toolkit-exploration/
---

Think of this work as a deep dive into the practical world of machine learning. Across several projects, I put key algorithms to the test using industry-standard tools like **Scikit-learn** for core models, **Matplotlib** for visualization, and **Gymnasium (Gym)** for reinforcement learning environments. The goal was figuring out *what* works best, *when*, and *why*.

Here's a glimpse:

### Predicting Outcomes (Supervised Learning with Scikit-learn)

*   Tackled real-world prediction challenges using datasets like the noisy **Titanic** survival records and structured **Date Fruit** classifications.
*   Compared standard **Scikit-learn** implementations of algorithms like Decision Trees, Neural Networks, SVM, KNN, and Boosting. Fine-tuning involved techniques like pruning for Decision Trees (comparing Gini/Entropy criteria) and optimizing network architecture/solvers for Neural Networks.
    *   *Example:* Analyzing Decision Tree pruning by comparing nodes removed using Gini vs. Entropy criteria.

        ![Comparison of nodes pruned in Decision Trees using Gini vs Entropy criteria](/images/ml-toolkit-exploration/node_cuted_gini_entropy.png)

    *   *Example:* Visualizing learning curves helps diagnose bias and variance issues during model training.

        ![Example Learning Curves showing Bias vs Variance](/images/ml-toolkit-exploration/learning_curves.png)

*   *Key Insight:* No single winner! **SVM** shone on the clean, structured Date Fruit data, while **KNN** proved surprisingly robust for the preprocessed, messier Titanic data. Success hinges on matching the algorithm and its tuning to the data's characteristics and noise level.

### Finding the Best Solutions (Randomized Optimization - Custom Library)

*   Went beyond standard libraries, implementing and comparing **Genetic Algorithms (GA), Simulated Annealing (SA),** and **MIMIC** using a **personalized optimization library**.
*   Applied these to classic problems like the Traveling Salesman (TSP), N-Queens, and FlipFlop. Tuning involved parameters like population size, mutation rate (GA), keep percentage (MIMIC), and temperature/decay schedules (SA).
    *   *Example:* GA tuning for TSP involved balancing population size and mutation rate for effective exploration vs. exploitation.

        ![Fine Tuning of the GA mutation Rate for TSP](/images/ml-toolkit-exploration/mut_rateGA.png)

    *   *Example:* MIMIC's performance on FlipFlop highlighted its strength in modeling structured problems, achieving optimal solutions efficiently by tuning population size and keep percentage.

        ![Fine tuning MIMIC Population Size and Keep Percentage for FlipFlop](/images/ml-toolkit-exploration/heatmapLargeMIMIC.png)

*   *Cool Result:* Showed **Simulated Annealing** could effectively optimize Neural Network weights for the Titanic dataset, sometimes outperforming traditional methods like gradient descent for specific setups.

### Uncovering Patterns & Simplifying Data (Unsupervised Learning with Scikit-learn)

*   Used **Scikit-learn's** K-Medoids and Expectation-Maximization (EM with Gaussian Mixture Models) algorithms to automatically discover hidden groups in both Titanic and Date Fruit datasets.
*   Leveraged dimensionality reduction techniques like Principal Component Analysis (PCA), Independent Component Analysis (ICA), Randomized Projections (RP), and Uniform Manifold Approximation and Projection (UMAP) to simplify data and enable visualization with **Matplotlib**.
    *   *Example:* ICA aims to find independent source signals, represented by the [ICA Mixing Matrix](/images/ml-toolkit-exploration/ICA_Mixing_Matrix.pdf). (Link to PDF)

*   *Highlight:* Dimensionality reduction techniques like UMAP were particularly effective at simplifying high-dimensional data while preserving meaningful structures.

### Teaching Agents to Decide (Reinforcement Learning with Gymnasium)

*   Built agents that learned optimal strategies in environments from the **Gymnasium (Gym)** library, like the slippery **Frozen Lake** (discrete states/actions) and the continuous **Mountain Car**.
    *   *Example:* Understanding the Mountain Car environment dynamics is key to solving it.

        ![Explanation of the Mountain Car environment state space](/images/ml-toolkit-exploration/explaination_mountain_car.png)

*   Implemented and compared core RL methods like Value Iteration, Policy Iteration, and Q-Learning. Analyzed the impact of hyperparameters like discount factor (gamma).
    *   *Example:* Comparing the optimal policies found by Value Iteration and Policy Iteration on Frozen Lake.

        ![Comparison of optimal policy maps from Value Iteration and Policy Iteration on Frozen Lake](/images/ml-toolkit-exploration/it_policy_map_vi.png)

    *   *Example:* Policy Iteration performance comparison on Frozen Lake with varying discount factors (gamma), showing how it affects the learned policy.

        ![Policy Iteration optimal policy maps vs. gamma for Frozen Lake](/images/ml-toolkit-exploration/policy_according_to_gamma.png)

*   *Learning:* Demonstrated how environment characteristics and hyperparameter tuning (like gamma affecting long-term reward focus) drastically influence the effectiveness and convergence of different RL algorithms.

**Overall:** This work demonstrates hands-on experience across the ML spectrum using key libraries (**Scikit-learn, Matplotlib, Gymnasium**) and custom implementations. It showcases the ability to select, implement, tune, and critically evaluate the right algorithms for diverse data types and challenges, from prediction and optimization to pattern discovery and sequential decision-making.

**Key Technologies:** **Python | Scikit-learn | Matplotlib | Gymnasium (Gym) | NumPy | Pandas | Randomized Optimization (GA, SA, MIMIC) | Supervised Learning (Decision Trees, NN, SVM, KNN, Boosting) | Unsupervised Learning (K-Medoids, EM, PCA, ICA, RP, UMAP) | Reinforcement Learning (Value Iteration, Policy Iteration, Q-Learning)**