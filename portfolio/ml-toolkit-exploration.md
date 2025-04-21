---
title: "Project Spotlight: Mastering the Machine Learning Toolkit"
excerpt: "A deep dive into supervised, unsupervised, randomized optimization, and reinforcement learning algorithms using Scikit-learn, Matplotlib, Gymnasium, and custom libraries."
collection: portfolio
header:
  teaser: "images/ml-toolkit-exploration/umap_fruits_clusters.png" # Example teaser image
permalink: /portfolio/ml-toolkit-exploration/
---

Think of this work as a deep dive into the practical world of machine learning. Across several projects, I put key algorithms to the test using industry-standard tools like **Scikit-learn** for core models, **Matplotlib** for visualization, and **Gymnasium (Gym)** for reinforcement learning environments. The goal was figuring out *what* works best, *when*, and *why*.

Here's a glimpse:

### Predicting Outcomes (Supervised Learning with Scikit-learn)

*   Tackled real-world prediction challenges using datasets like the noisy **Titanic** survival records and structured **Date Fruit** classifications.
*   Compared standard **Scikit-learn** implementations of algorithms like Decision Trees, Neural Networks, SVM, KNN, and Boosting. Fine-tuning involved techniques like pruning for Decision Trees (comparing Gini/Entropy criteria and `ccp_alpha`) and optimizing network architecture/solvers for Neural Networks.
    *   *Example:* Tuning `ccp_alpha` for Decision Trees on the Fruits dataset showed how post-pruning retains useful deep branches compared to simple `max_depth` limitation.

        ![Fine tuning using max_depth (a), ccp_alpha (b) for Decision Trees on Date Fruit dataset](/images/ml-toolkit-exploration/tree_fruits_ccp_alpha_tuning.png)

    *   *Example:* Neural Network architecture search revealed optimal layer/neuron counts for different datasets.

        ![Cross validation F1 score vs. NN architecture (layers, neurons) for Titanic dataset](/images/ml-toolkit-exploration/nn_titanic_architerture.png)

    *   *Example:* Comparing SVM kernels and tuning C/Gamma highlighted the trade-offs for different data structures.

        ![SVM F1 score heatmap vs. C and Gamma (RBF kernel) for Titanic dataset](/images/ml-toolkit-exploration/svm_titanic_heat_map_rbf.png)

    *   *Example:* Boosting (AdaBoost with Decision Tree base estimators) performance analysis on the Date Fruit dataset, showing the impact of the number of estimators.

        ![Boosting (AdaBoost) F1 score vs. number of estimators for Date Fruit dataset](/images/ml-toolkit-exploration/boosting_fruits_estimators.png)

*   *Key Insight:* No single winner! **SVM** shone on the clean, structured Date Fruit data (achieving 0.933 F1), while **KNN** proved surprisingly robust for the preprocessed, messier Titanic data (0.822 F1). Success hinges on matching the algorithm and its tuning (like distance metrics for KNN or kernels for SVM) to the data's characteristics and noise level.

### Finding the Best Solutions (Randomized Optimization - Custom Library)

*   Went beyond standard libraries, implementing and comparing **Genetic Algorithms (GA), Simulated Annealing (SA),** and **MIMIC** using a **personalized optimization library**.
*   Applied these to classic problems like the Traveling Salesman (TSP), N-Queens, and FlipFlop. Tuning involved parameters like population size, mutation rate (GA), keep percentage (MIMIC), and temperature/decay schedules (SA).
    *   *Example:* GA tuning for TSP involved balancing population size and mutation rate for effective exploration vs. exploitation.

        ![Fine Tuning of the GA mutation Rate for TSP](/images/ml-toolkit-exploration/tsp_ga_mutation_rate.png)

    *   *Example:* MIMIC's performance on FlipFlop highlighted its strength in modeling structured problems, achieving optimal solutions efficiently.

        ![Fine tuning MIMIC Population Size and Keep Percentage for FlipFlop](/images/ml-toolkit-exploration/FlipFlop_heatmapLargeMIMIC.png)

    *   *Example:* SA tuning for the N-Queens problem showed its ability to escape local optima in vast search spaces by adjusting temperature and decay.

        ![N-Queens Fitness vs. SA initial temperature and decay rate](/images/ml-toolkit-exploration/Queens_SAinittempQueens.png)

*   *Cool Result:* Showed **Simulated Annealing** could effectively optimize Neural Network weights for the Titanic dataset, sometimes outperforming traditional methods like gradient descent for specific setups, though convergence behavior needed careful monitoring.

    ![Learning Curve of SA for NN weight optimization on Titanic](/images/ml-toolkit-exploration/NN_lcSA.png)

### Uncovering Patterns & Simplifying Data (Unsupervised Learning with Scikit-learn)

*   Used **Scikit-learn's** K-Medoids and Expectation-Maximization (EM with Gaussian Mixture Models) algorithms to automatically discover hidden groups in both Titanic and Date Fruit datasets. Evaluated cluster quality using metrics like Silhouette Score and BIC/AIC.
    *   *Example:* Comparing K-Medoids and EM clustering performance on the Date Fruit dataset using Silhouette scores.

        ![Silhouette scores for K-Medoids and EM clustering on Date Fruit dataset](/images/ml-toolkit-exploration/clustering_fruits_silhouette.png)

    *   *Example:* Visualizing clusters found by EM on the Titanic dataset (after dimensionality reduction).

        ![EM clustering results visualized on reduced Titanic dataset](/images/ml-toolkit-exploration/em_titanic_clusters.png)

*   Leveraged dimensionality reduction techniques like Principal Component Analysis (PCA), Independent Component Analysis (ICA), Randomized Projections (RP), and Uniform Manifold Approximation and Projection (UMAP) to simplify data and enable visualization with **Matplotlib**.
    *   *Example:* PCA explained variance ratio for the Date Fruit dataset, helping determine the optimal number of components.

        ![PCA Explained Variance Ratio for Date Fruit dataset](/images/ml-toolkit-exploration/pca_fruits_explained_variance.png)

    *   *Example:* Comparing dimensionality reduction techniques (PCA, ICA, RP, UMAP) for visualizing Date Fruit clusters.

        ![Date Fruit data projected onto 2D using PCA, ICA, RP, and UMAP](/images/ml-toolkit-exploration/dim_reduction_fruits_comparison.png)

*   *Highlight:* **UMAP** was particularly effective at simplifying high-dimensional data while preserving meaningful structures, revealing clear clusters in the Date Fruit dataset that were less distinct with other methods.

    ![UMAP visualization of Date Fruit clusters](/images/ml-toolkit-exploration/umap_fruits_clusters.png)

### Teaching Agents to Decide (Reinforcement Learning with Gymnasium)

*   Built agents that learned optimal strategies in environments from the **Gymnasium (Gym)** library, like the slippery **Frozen Lake** (discrete states/actions) and the continuous **Mountain Car**.
*   Implemented and compared core RL methods like Value Iteration, Policy Iteration, and Q-Learning. Analyzed the impact of hyperparameters like discount factor (gamma), learning rate (alpha), and exploration rate (epsilon).
    *   *Example:* Value Iteration convergence on the Frozen Lake environment, showing the value function stabilizing over iterations.

        ![Value Iteration convergence plot for Frozen Lake (8x8)](/images/ml-toolkit-exploration/vi_frozenlake_convergence.png)

    *   *Example:* Policy Iteration performance comparison on Frozen Lake with varying discount factors (gamma).

        ![Policy Iteration performance (mean reward) vs. gamma for Frozen Lake](/images/ml-toolkit-exploration/pi_frozenlake_gamma.png)

    *   *Example:* Q-Learning performance on Frozen Lake, illustrating the learning curve (average reward per episode) and the effect of exploration decay.

        ![Q-Learning average reward per episode for Frozen Lake (8x8)](/images/ml-toolkit-exploration/qlearning_frozenlake_reward.png)

    *   *Example:* Q-Learning tackling the Mountain Car problem, showing reward improvement over episodes.

        ![Q-Learning average reward per episode for Mountain Car](/images/ml-toolkit-exploration/qlearning_mountaincar_reward.png)

*   *Learning:* Demonstrated how environment characteristics (stochasticity in Frozen Lake vs. continuous states in Mountain Car) and hyperparameter tuning (gamma affecting long-term reward focus, epsilon balancing exploration/exploitation) drastically influence the effectiveness and convergence of different RL algorithms.

**Overall:** This work demonstrates hands-on experience across the ML spectrum using key libraries (**Scikit-learn, Matplotlib, Gymnasium**) and custom implementations. It showcases the ability to select, implement, tune, and critically evaluate the right algorithms for diverse data types and challenges, from prediction and optimization to pattern discovery and sequential decision-making.

**Key Technologies:** **Python | Scikit-learn | Matplotlib | Gymnasium (Gym) | NumPy | Pandas | Randomized Optimization (GA, SA, MIMIC) | Supervised Learning (Decision Trees, NN, SVM, KNN, Boosting) | Unsupervised Learning (K-Medoids, EM, PCA, ICA, RP, UMAP) | Reinforcement Learning (Value Iteration, Policy Iteration, Q-Learning)**