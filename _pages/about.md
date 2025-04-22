---
permalink: /
title: "" # Add a title if you want one, or leave blank
excerpt: "About me"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

## Projects

Here are some of the projects I've worked on. Click on them to learn more!

<div class="portfolio-list">
{% assign desired_order = "modernpatentbert,backdooring-agent.md,pytorch-journey,deepip-patent-ai,llm-ranking-framework,ml-toolkit-exploration,lingua-custodia-pipelines,hackathon-language-assistant" | split: ',' %}
{% assign projects_in_order = "" | split: "" %}
{% for slug in desired_order %}
  {% assign project = site.portfolio | where: "slug", slug | first %}
  {% if project %}
    {% assign projects_in_order = projects_in_order | push: project %}
  {% endif %}
{% endfor %}

{% for post in projects_in_order %}
  <div class="portfolio-item" style="display: flex; align-items: center; gap: 1em; margin-bottom: 1em;">
    {% if post.header.teaser %}
      <div class="portfolio-item-teaser" style="flex-shrink: 0;">
        <a href="{{ post.url | relative_url }}">
          <img src="{{ post.header.teaser | relative_url }}" alt="{{ post.title }} preview" style="max-width: 150px; display: block;">
        </a>
      </div>
    {% endif %}
    <div class="portfolio-item-content">
      <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
      {% if post.excerpt %}
        <p>{{ post.excerpt | strip_html | truncatewords: 30 }}</p>
      {% endif %}
    </div>
  </div>
{% endfor %}
</div>

## Education

*   **Georgia Institute of Technology** (2024 – 2025)
    *   MS in Computer Science
*   **Université Technologique de Compiègne** (2019 – 2025)
    *   BS & MEng in Computer Science
*   **Relevant Coursework:** Deep Learning, Machine Learning, Machine Learning Security, NLP.

## Experience

*   **Researcher Assistant** supervised by **[Prof. Teodora Baluta](https://teobaluta.github.io/)**, Georgia Tech (Jan. - May 2025)
    *   Watermarking method for LLMs resilient to robust aggregators in a Federated Learning setting.
    *   Novel Backdooring method for LLM Code Act Agent with 99% Attack Success Rate.
*   **Machine Learning Engineer Intern**, Kili Technology (DeepIP filiale) -- Paris (Jun. - Dec. 2024)
    *   Developed a lightweight, database-less patent similarity retrieval pipeline using a novel combination of LLMs, search APIs, and embedding ranking, enabling deployment in resource-constrained environments.
    *   Engineered an end-to-end Style Transfer system for patent section generation, incorporating:
        *   Custom fine-tuning dataset creation using advanced NLP techniques, clustering analysis & LLM as a Judge.
        *   Enhanced generation quality by integrating Chain of Thought reasoning, experimenting with DPSy.
*   **Data Intern**, Lingua Custodia -- Paris (Feb. – Jul. 2023)
    *   Developed a [Python package](https://gitlab.com/linguacustodia/easylaser) for efficient embedding generation with multi-GPU support.
    *   Implemented cross-language sentence similarity analysis and benchmarking for machine translation applications.
    *   Architected scalable data pipelines for preprocessing and cleaning machine translation training datasets.
    *   Designed and deployed a comprehensive scraping framework integrated with database and datalake systems.

## Technologies

*   **Languages:** Python (PyTorch, NumPy, Scikit-learn, Transformers, Multithreading, Gym, Selenium), Go, R, SQL.
*   **Databases:** SQL (PostgreSQL, SQLite, EdgeDB).
*   **Tools:** Git, Docker, Bash, Weights & Biases, DeepSpeed, VLLM, Ollama, Unsloth, Google Cloud.
*   **OS:** macOS, Ubuntu, Windows (WSL).