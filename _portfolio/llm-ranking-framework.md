---
title: "Ethical AI Recommendations: Benchmarking LLM Bias in Cold-Start Scenarios"
excerpt: "Developed and applied a novel benchmark to evaluate ethical biases (gender, nationality, etc.) in LLM-based recommender systems, especially for new users (cold-start), revealing significant stereotype replication and providing tools for fairer AI."
collection: portfolio
header:
  teaser: "/images/llm-rec-bias/pipeline.png" # Using your pipeline diagram as the teaser
# permalink: /portfolio/llm-rec-bias/ # Optional: uncomment and adjust if needed
---

Large Language Models (LLMs) like ChatGPT, Llama, and Gemma are rapidly entering new domains, including **recommender systems**. Imagine asking an AI chatbot, "What movie should I watch?" or "Which college should I consider?" The promise is powerful: personalized suggestions leveraging the LLM's vast knowledge.

This works reasonably well when the AI knows a lot about you. But what about **new users** – the "cold-start" problem? With little to no interaction history, how does the LLM make recommendations?

### The Hidden Risk: Bias from Scarcity

In cold-start scenarios, LLMs might grasp at the only straws available: basic demographic details provided by the user, often including **sensitive attributes** like gender, nationality, age, or race. Relying heavily on these attributes is ethically dangerous.

*   **Stereotyping:** The LLM might recommend action movies primarily to boys and romance movies to girls, reinforcing societal stereotypes. [Figure 7](#figure-7)
*   **Limited Exposure:** Users might be pigeonholed early on, missing out on diverse content or opportunities simply because of their demographic profile.
*   **High-Stakes Impact:** While biased movie recommendations are problematic, biased *college* recommendations could have serious, life-altering consequences, potentially limiting educational paths based on sensitive attributes.

Existing benchmarks often evaluated LLMs with *some* user context (like a favorite artist), not the true zero-context cold-start situation. They also lacked flexibility.

### Our Approach: A Dedicated Cold-Start Bias Benchmark

To address this gap, we developed a new benchmark framework specifically designed to evaluate LLM bias in **zero-context, attribute-only recommendation scenarios.**

**Core Idea: Counterfactual Probing**

1.  Ask the LLM to recommend items (movies, songs, colleges) for a **neutral user** (no attributes specified).
2.  Ask the LLM to recommend items for the *same task* but specify **only one sensitive attribute** (e.g., "Recommend for a girl," "Recommend for a Chinese user").
3.  **Compare the lists.** How much do the recommendations change *just* because of that single sensitive attribute?

<!-- Your Pipeline Diagram -->
<p style="text-align: center;">
  <img id="figure-1" src="/images/llm-rec-bias/pipeline.png" alt="Diagram showing the benchmark pipeline: Datasets + Attributes -> Prompts -> LLM -> Output -> Bias Scores" style="max-width: 100%; height: auto; display: block; margin-left: auto; margin-right: auto;">
</p>
<p style="text-align: center;">
  <em>Figure 1: Our automated benchmark pipeline generates prompts with/without sensitive attributes, gets LLM recommendations, and calculates bias scores based on how much the recommendations differ.</em>
</p>

**Key Features of Our Benchmark:**

*   **Cold-Start Focus:** Simulates scenarios where only sensitive attributes are known.
*   **Modular & Automated:** Easily integrates different datasets (we added a College dataset alongside Music/Movies), sensitive attributes, and open-source LLMs (thanks to vLLM).
*   **Robust Metrics:** Uses established list comparison metrics (IOU/Jaccard, SERP, PRAG) to quantify the difference ("Divergence" or "Bias") between neutral and attribute-specific recommendations. Higher divergence means the attribute had a bigger impact, indicating potential bias.
*   **Reproducibility:** Provides item lists and code for others to use.

### Uncovering Biases: Key Findings from Gemma & Llama

We tested several models, primarily focusing on Google's Gemma 3 family (1B, 4B, 12B parameters) and Meta's Llama 3.2 3B, across Movie, Music, and College recommendation tasks (re-ranking a list of 500 items).

**Finding 1: Bigger Isn't Always Less Biased (H2)**

Does model size correlate with fairness? Not necessarily. We found a non-monotonic relationship.

*   The **Gemma 3 4B** model often showed the *lowest* divergence (least bias/sensitivity) across datasets compared to the 1B and 12B models.
*   The **Gemma 3 12B** model, while capable, was often *more sensitive* to attributes, leading to higher divergence. It seemed to amplify biases sometimes.
*   The **Gemma 3 1B** model struggled with the task itself, leading to high divergence due to inconsistent outputs.

<!-- Overall Mean IOU Divergence per Model Across Datasets -->
<p style="text-align: center;">
  <img id="figure-2" src="/images/llm-rec-bias/overall_iou_gemma_sizes.png" alt="Bar chart showing Overall Mean IOU Divergence for Gemma 1B, 4B, 12B across College, Music, Movie datasets" style="max-width: 80%; height: auto; display: block; margin-left: auto; margin-right: auto;">
</p>
<p style="text-align: center;">
  <em>Figure 2: Gemma 3 4B (green) often showed the lowest average bias (IOU Divergence) compared to 1B (blue) and 12B (orange), especially on Music and Movie datasets. (Figure adapted from project report)</em>
</p>

*   **Takeaway:** The mid-size model (4B) hit a sweet spot between task competence and lower sensitivity to sensitive attributes. Simply scaling up doesn't guarantee fairness.

**Finding 2: Societal Stereotypes are Mirrored & Amplified (H3)**

LLMs learn from text data, which contains societal biases. Do they replicate them? Yes.

*   **Gender & Movies:** When recommending movies, Gemma models strongly associated action movies with boys/males and steered away from them for girls/females.
*   **Amplification:** This stereotyping was *more extreme* in the larger Gemma 3 12B model (0% action movies recommended for 'a girl') than the 4B model.

<!-- Action Movie Bias - Gemma 3 12B -->
<p style="text-align: center;">
  <img id="figure-3" src="/images/llm-rec-bias/action_movie_bias_12B.png" alt="Bar chart showing percentage of action movies recommended by Gemma 3 12B for different gender personas" style="max-width: 70%; height: auto; display: block; margin-left: auto; margin-right: auto;">
</p>
<p style="text-align: center;">
  <em>Figure 3: Strong gender stereotyping in Gemma 3 12B movie recommendations. 'A boy' gets >60% action movies, while 'a girl' gets 0%. (Figure adapted from project report)</em>
</p>

*   **Takeaway:** LLMs readily absorb and deploy common gender stereotypes in recommendations, and larger models might even intensify these biases without specific mitigation.

**Finding 3: Relevant Context Can Mitigate Bias (H4)**

If we give the LLM *relevant* preference information alongside a sensitive attribute, does it reduce reliance on the attribute? Yes.

*   We tested Gemma 3 12B by adding the context "who is an action movie fan" to the prompts.
*   The divergence scores (bias) decreased significantly across most sensitive attributes (gender, race, occupation, etc.) when this context was present. The model prioritized the stated preference over the sensitive attribute.

<!-- Spider Plot Showing Context Mitigation -->
<p style="text-align: center;">
  <img id="figure-4" src="/images/llm-rec-bias/context_mitigation_spider.png" alt="Spider plot comparing IOU divergence with (orange) and without (blue) action movie fan context across various attributes" style="max-width: 90%; height: auto; display: block; margin-left: auto; margin-right: auto;">
</p>
<p style="text-align: center;">
  <em>Figure 4: Adding context ('action movie fan', orange line) significantly reduced bias (IOU Divergence) compared to no context (blue line) across many sensitive attributes. The orange shape is closer to the center (lower divergence). (Figure adapted from project report)</em>
</p>

*   **Takeaway:** Providing genuine user preferences is a powerful way to steer the LLM away from relying on potentially biased demographic assumptions. This highlights the importance of gathering user feedback over time.

**Finding 4: A Strong Default Bias Towards Western Content (H5)**

Do LLMs favor content from certain cultures? Our findings suggest a strong Western bias.

*   For a **neutral user**, Gemma 3 12B recommended movie lists that were overwhelmingly **~91% Western** (North American, European, etc.).
*   While prompts with non-Western attributes (e.g., 'a Chinese user', 'an African user') did lead to *more* non-Western recommendations compared to the neutral baseline, the lists often remained dominated by Western content. For example, 'an African user' still received 96% Western movies.

<!-- Western Bias Bar Chart -->
<p style="text-align: center;">
  <img id="figure-5" src="/images/llm-rec-bias/western_bias_bar.png" alt="Bar chart showing percentage of Western items recommended for different attributes" style="max-width: 90%; height: auto; display: block; margin-left: auto; margin-right: auto;">
</p>
<p style="text-align: center;">
  <em><a id="figure-5"></a>Figure 5: High percentage of Western movies recommended even for non-Western personas. The neutral baseline is already ~91% Western. (Figure adapted from project report)</em>
</p>

*   **Takeaway:** The models tested seem to have a strong default towards Western culture, likely reflecting imbalances in their training data. This could lead to culturally narrow recommendations unless addressed.

**(Other findings:** We also noted that Llama 3.2 3B showed instability issues, making Gemma a more reliable choice for this specific bias analysis. We also found that instruction-tuning is necessary for these models to even perform the ranking task, preventing a direct bias comparison with non-tuned base models using this framework.)

<!-- Optional: Detailed Metric Scores Plot -->
<p style="text-align: center;">
  <img id="figure-6" src="/images/llm-rec-bias/metric_scores_gemma12b_movie.png" alt="Detailed bar chart showing IOU, Pragmatic, SERP MS Divergence scores for Gemma 3 12B on Movie dataset across all attributes" style="max-width: 100%; height: auto; display: block; margin-left: auto; margin-right: auto;">
</p>
<p style="text-align: center;">
  <em><a id="figure-6"></a>Figure 6: Detailed divergence scores across attributes for Gemma 3 12B (Movie dataset), showing variation in bias depending on the attribute and the metric used. (Figure adapted from project report)</em>
</p>


### Why This Project Matters & Conclusion

As LLMs become integrated into recommendation platforms, understanding and mitigating their potential biases is crucial for fairness and user trust. This is especially true in cold-start situations where the risk of relying on sensitive attributes is highest.

**Our contribution is a dedicated benchmark and automated pipeline to systematically evaluate these biases.**

**Key Takeaways:**

1.  **Bias is Real:** Current LLMs exhibit significant biases based on gender, nationality, and culture, especially in cold-start scenarios.
2.  **Size Isn't a Silver Bullet:** Larger models aren't inherently fairer; mid-sized models might offer a better balance.
3.  **Context Matters:** Providing relevant user preferences effectively reduces reliance on sensitive attributes.
4.  **Tooling is Key:** Our benchmark provides a practical tool for developers to assess and compare bias in different LLMs *before* deployment.
5.  **Ethical Responsibility:** Platforms using LLMs for recommendations, particularly for high-stakes decisions like college choice, must actively evaluate and mitigate potential biases.

By enabling easier detection of these issues, we hope to encourage the development of more ethically aligned and fair AI recommender systems.

*This project was a collaborative effort with [Alexandre ANDRE](https://www.linkedin.com/in/alexandre-andre-/).*

---

**Code Repository:** [**https://github.com/GauthierRoy/biais_llm_rec**](https://github.com/GauthierRoy/biais_llm_rec)

### Key Technologies & Concepts

*   **Core Libraries:** Python, PyTorch, Hugging Face (Transformers), vLLM (for inference), Pandas, Matplotlib/Seaborn.
*   **Models Tested:** Gemma 3 (1B, 4B, 12B instruction-tuned), Llama 3.2 (3B instruction-tuned).
*   **Techniques:** Recommender Systems (Cold-Start, Re-ranking), LLM Prompting, Counterfactual Evaluation, Bias Measurement.
*   **Metrics:** IOU (Jaccard) Divergence, SERP-MS Divergence, Pragmatic Divergence.
*   **Datasets:** Movie (TMDB), Music (Spotify), College (QS Rankings).
*   **Concepts:** Ethical AI, Algorithmic Bias, Fairness in Machine Learning, Sensitive Attributes, Stereotyping.

---