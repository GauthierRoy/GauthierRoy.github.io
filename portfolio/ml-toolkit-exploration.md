**Project Spotlight: Mastering the Machine Learning Toolkit**

Think of this work as a deep dive into the practical world of machine learning. Across several projects, I put key algorithms to the test using industry-standard tools like **Scikit-learn** for core models, **Matplotlib** for visualization, and **Gymnasium (Gym)** for reinforcement learning environments. The goal was figuring out *what* works best, *when*, and *why*.

Here's a glimpse:

*   **Predicting Outcomes (Supervised Learning with Scikit-learn):**
    *   Tackled real-world prediction challenges using datasets like the noisy **Titanic** survival records and structured **Date Fruit** classifications.
    *   Compared standard **Scikit-learn** implementations of algorithms like Decision Trees, Neural Networks, SVM, KNN, and Boosting. Fine-tuning involved techniques like pruning for Decision Trees (comparing Gini/Entropy criteria and `ccp_alpha`) and optimizing network architecture/solvers for Neural Networks.
        *   *Example:* Tuning `ccp_alpha` for Decision Trees on the Fruits dataset showed how post-pruning retains useful deep branches compared to simple `max_depth` limitation.
            ```
            ![Fine tuning using max_depth (a), ccp_alpha (b)](/images/ml-toolkit-exploration/tree_fruits_ccp_alpha_tuning.png)
            ```
        *   *Example:* Neural Network architecture search revealed optimal layer/neuron counts for different datasets.
            ```
            ![Cross validation F1 score in relation to number of neurons per layer on (y-axis) and number of layers (x-axis) for Titanic](/images/ml-toolkit-exploration/nn_titanic_architerture.png)
            ```
        *   *Example:* Comparing SVM kernels and tuning C/Gamma highlighted the trade-offs for different data structures.
            ```
            ![F1 score in relation to C and Gamma with RBF kernel for Titanic](/images/ml-toolkit-exploration/svm_titanic_heat_map_rbf.png)
            ```
    *   *Key Insight:* No single winner! **SVM** shone on the clean, structured Date Fruit data (achieving 0.933 F1), while **KNN** proved surprisingly robust for the preprocessed, messier Titanic data (0.822 F1). Success hinges on matching the algorithm and its tuning (like distance metrics for KNN or kernels for SVM) to the data's characteristics and noise level.

*   **Finding the Best Solutions (Randomized Optimization - Custom Library):**
    *   Went beyond standard libraries, implementing and comparing **Genetic Algorithms (GA), Simulated Annealing (SA),** and **MIMIC** using a **personalized optimization library**.
    *   Applied these to classic problems like the Traveling Salesman (TSP), N-Queens, and FlipFlop. Tuning involved parameters like population size, mutation rate (GA), keep percentage (MIMIC), and temperature/decay schedules (SA).
        *   *Example:* GA tuning for TSP involved balancing population size and mutation rate for effective exploration vs. exploitation.
            ```
            ![Fine Tuning of the GA mutation Rate for TSP](/images/ml-toolkit-exploration/tsp_ga_mutation_rate.png)
            ```
        *   *Example:* MIMIC's performance on FlipFlop highlighted its strength in modeling structured problems, achieving optimal solutions efficiently.
            ```
            ![Fine tunning of the MIMIC Population Size and mutation Rate for FlipFlop](/images/ml-toolkit-exploration/FlipFlop_heatmapLargeMIMIC.png)
            ```
        *   *Example:* SA tuning for the N-Queens problem showed its ability to escape local optima in vast search spaces.
            ```
            ![Fitness variation with different initial temperatures and decay rates for SA on Queens](/images/ml-toolkit-exploration/Queens_SAinittempQueens.png)
            ```
    *   *Cool Result:* Showed **Simulated Annealing** could effectively optimize Neural Network weights for the Titanic dataset, sometimes outperforming traditional methods like gradient descent for specific setups, though convergence behavior needed careful monitoring.
        ```
        ![Learning Curve of SA for NN weight optimization on Titanic](/images/ml-toolkit-exploration/NN_lcSA.png)
        ```

*   **Uncovering Patterns & Simplifying Data (Unsupervised Learning with Scikit-learn):**
    *   Used **Scikit-learn's** K-Medoids and Expectation-Maximization (EM with Gaussian Mixture Models) algorithms to automatically discover hidden groups in both Titanic and Date Fruit<!-- filepath: c:\Users\gauth\OneDrive\Desktop\GauthierRoy.github.io\portfolio\ml-toolkit-exploration.md -->
**Project Spotlight: Mastering the Machine Learning Toolkit**

Think of this work as a deep dive into the practical world of machine learning. Across several projects, I put key algorithms to the test using industry-standard tools like **Scikit-learn** for core models, **Matplotlib** for visualization, and **Gymnasium (Gym)** for reinforcement learning environments. The goal was figuring out *what* works best, *when*, and *why*.

Here's a glimpse:

*   **Predicting Outcomes (Supervised Learning with Scikit-learn):**
    *   Tackled real-world prediction challenges using datasets like the noisy **Titanic** survival records and structured **Date Fruit** classifications.
    *   Compared standard **Scikit-learn** implementations of algorithms like Decision Trees, Neural Networks, SVM, KNN, and Boosting. Fine-tuning involved techniques like pruning for Decision Trees (comparing Gini/Entropy criteria and `ccp_alpha`) and optimizing network architecture/solvers for Neural Networks.
        *   *Example:* Tuning `ccp_alpha` for Decision Trees on the Fruits dataset showed how post-pruning retains useful deep branches compared to simple `max_depth` limitation.
            ```
            ![Fine tuning using max_depth (a), ccp_alpha (b)](/images/ml-toolkit-exploration/tree_fruits_ccp_alpha_tuning.png)
            ```
        *   *Example:* Neural Network architecture search revealed optimal layer/neuron counts for different datasets.
            ```
            ![Cross validation F1 score in relation to number of neurons per layer on (y-axis) and number of layers (x-axis) for Titanic](/images/ml-toolkit-exploration/nn_titanic_architerture.png)
            ```
        *   *Example:* Comparing SVM kernels and tuning C/Gamma highlighted the trade-offs for different data structures.
            ```
            ![F1 score in relation to C and Gamma with RBF kernel for Titanic](/images/ml-toolkit-exploration/svm_titanic_heat_map_rbf.png)
            ```
    *   *Key Insight:* No single winner! **SVM** shone on the clean, structured Date Fruit data (achieving 0.933 F1), while **KNN** proved surprisingly robust for the preprocessed, messier Titanic data (0.822 F1). Success hinges on matching the algorithm and its tuning (like distance metrics for KNN or kernels for SVM) to the data's characteristics and noise level.

*   **Finding the Best Solutions (Randomized Optimization - Custom Library):**
    *   Went beyond standard libraries, implementing and comparing **Genetic Algorithms (GA), Simulated Annealing (SA),** and **MIMIC** using a **personalized optimization library**.
    *   Applied these to classic problems like the Traveling Salesman (TSP), N-Queens, and FlipFlop. Tuning involved parameters like population size, mutation rate (GA), keep percentage (MIMIC), and temperature/decay schedules (SA).
        *   *Example:* GA tuning for TSP involved balancing population size and mutation rate for effective exploration vs. exploitation.
            ```
            ![Fine Tuning of the GA mutation Rate for TSP](/images/ml-toolkit-exploration/tsp_ga_mutation_rate.png)
            ```
        *   *Example:* MIMIC's performance on FlipFlop highlighted its strength in modeling structured problems, achieving optimal solutions efficiently.
            ```
            ![Fine tunning of the MIMIC Population Size and mutation Rate for FlipFlop](/images/ml-toolkit-exploration/FlipFlop_heatmapLargeMIMIC.png)
            ```
        *   *Example:* SA tuning for the N-Queens problem showed its ability to escape local optima in vast search spaces.
            ```
            ![Fitness variation with different initial temperatures and decay rates for SA on Queens](/images/ml-toolkit-exploration/Queens_SAinittempQueens.png)
            ```
    *   *Cool Result:* Showed **Simulated Annealing** could effectively optimize Neural Network weights for the Titanic dataset, sometimes outperforming traditional methods like gradient descent for specific setups, though convergence behavior needed careful monitoring.
        ```
        ![Learning Curve of SA for NN weight optimization on Titanic](/images/ml-toolkit-exploration/NN_lcSA.png)
        ```

*   **Uncovering Patterns & Simplifying Data (Unsupervised Learning with Scikit-learn):**
    *   Used **Scikit-learn's** K-Medoids and Expectation-Maximization (EM with Gaussian Mixture Models) algorithms to automatically discover hidden groups in both Titanic and Date Fruit