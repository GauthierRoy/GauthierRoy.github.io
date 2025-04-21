---
title: "Project Spotlight: NLP for Patent Search & Generation at DeepIP"
excerpt: "Developed and evaluated patent similarity search methods (Embeddings: OpenAI, FAISS, BERT/Google Patents POC, LLM+EPO POC). Specialized LLMs for patent generation via fine-tuning (Mistral, Gemini, OpenAI; Style Parameter analysis) and advanced instruction design (CoT, Few-Shot, DSPy evaluation). Integrated style transfer features via architectural refactoring (Onion Architecture)."
collection: portfolio
header:
  teaser: "/images/deepip-internship/slide4_poc_no_dataset.png" # Placeholder - Using Slide 4 right diagram from initial slides
# permalink: /portfolio/deepip-internship/
---

This project details the technical contributions during a six-month NLP Machine Learning Engineer internship at DeepIP (a Kili Technology solution). The work focused on enhancing patent similarity search and specializing Large Language Models (LLMs) for automated patent drafting tasks, involving research, implementation, evaluation, and integration into the existing codebase.

### Enhancing Similar Patent Search (Prior Art Discovery)

The primary goal was to develop methods for finding patents similar to a draft, crucial for prior art identification and technology monitoring.

**1. Embedding-Based Vector Search:**

*   **Core Implementation:** Built a pipeline using text embeddings and FAISS for efficient vector search.
    *   Patent text (initially tested with title and claims) is converted to embeddings.
    *   **Models Evaluated:** Initial tests used OpenAI API (`text-embedding-ada-002`), locally run open-source models via Hugging Face/PyTorch, and specialized models like PatentBERT. OpenAI's model was selected for performance in initial evaluations on a small, manually curated dataset.
    *   **Vector Indexing:** FAISS was used to index embeddings for fast Approximate Nearest Neighbor search, suitable for large datasets (estimated 6M+ patents). Techniques like partitioning and compression within FAISS were considered for scalability.
    *   **Challenges:** Explored the significant cost (~$2000+/month estimated) and maintenance overhead of hosting a large-scale vector database (e.g., using Lantern, Qdrant, or self-managed PostgreSQL with vector extensions). Quantization was considered but raises performance trade-offs.

    ![Core Embedding Search Workflow using OpenAI and Faiss](/images/deepip-internship/slide3_embedding_search_workflow.png) # Placeholder - Using Slide 3 diagram from initial slides

**2. Alternative Search Strategies (POCs & Explorations):**

*   **Leveraging Google Patents Dataset Embeddings:**
    *   **Concept:** Google provides pre-computed embeddings for patents, potentially allowing similarity search without generating/hosting them internally.
    *   **Challenge:** The embedding model used by Google is not public. Therefore, generating an embedding for a *new* patent draft to compare against the dataset is impossible directly.
    *   **POC Approach:** Explored training a linear layer to project embeddings from a known model (e.g., BERT-like) onto the Google embedding space. Required aligning a large number of BERT embeddings with corresponding Google embeddings. A test with 5000 embeddings was inconclusive, likely needing significantly more data or indicating divergence between the models.

    ![Custom BERT Embedding Projection Concept](/images/deepip-internship/slide4_bert_google_patent.png) # Placeholder - Using Slide 4 left diagram from initial slides

*   **Hybrid LLM & Classical Search Engine (EPO API):**
    *   **Concept:** Combine LLM intelligence with an external patent search API (like EPO's) to avoid hosting a full patent database.
    *   **POC Workflow:**
        1.  An LLM (specialized via fine-tuning or instruction design) analyzes a patent draft to generate relevant classical search query strings (e.g., Boolean queries with keywords and operators like `(car* OR vehicle) NEAR/6 driving NEAR/10 tires`).
        2.  The generated query is sent to the EPO API.
        3.  Top N results (e.g., 50-100) are retrieved.
        4.  Embeddings are calculated for the initial draft and the retrieved patents.
        5.  Retrieved patents are re-ranked based on embedding similarity to the draft. The top 5 are presented.
        6.  Logic included to potentially refine the query string via LLM if initial results have low similarity or too few results are returned.
    *   **Findings:** Encouraging results in finding thematically similar patents. However, limitations of classical keyword search (synonyms, phrasing) remain, reinforcing the theoretical advantage of pure vector search. Reliance on EPO API quality was also noted.

    ![Hybrid LLM and EPO Search POC Workflow](/images/deepip-internship/slide4_poc_no_dataset.png) # Placeholder - Using Slide 4 right diagram from initial slides

### Specializing Large Language Models (LLMs) for Patent Generation

Focused on improving the generation of specific patent sections (initially, the "Summary" section based on "Claims") by adapting LLMs to the patent domain's specific style and content requirements.

**1. Strategy: Fine-tuning vs. Advanced Instruction Design:**

*   Evaluated the trade-offs based on criteria relevant to DeepIP:
    *   **Cost:** Fine-tuning (training + hosting specialized model) generally higher than API calls for instruction-based approaches, though smaller fine-tuned models *could* be cheaper long-term.
    *   **Data Sovereignty:** Instruction design often relies on external APIs (like OpenAI/Azure), while fine-tuning allows using self-hosted or open-source models, offering more control. Azure deployments provided sufficient guarantees for DeepIP's client needs.
    *   **Deployment Complexity:** Fine-tuning involves model training, versioning, and hosting infrastructure. Instruction design is simpler, primarily involving API calls and logic.
    *   **Adaptability & Maintenance:** Instruction design is often faster to adapt; fine-tuning requires retraining for significant changes.
    *   **Performance:** Fine-tuning has higher *potential* performance ceiling for highly specific tasks; instruction design performance heavily depends on the base model and instruction quality.
*   **Decision:** **Instruction Design** was prioritized due to faster iteration, lower initial cost, sufficient performance with CoT, and adequate data security via Azure. Fine-tuning remained a valuable option explored for potential future use or smaller model deployment.

**2. LLM Fine-tuning Exploration (Summary Generation Task):**

*   **Data Collection & Preparation:**
    *   Developed a parser for USPTO patent data (XML format).
    *   Extracted "Claims" and "Summary" sections. Implemented rigorous filtering (e.g., removing patents with empty sections, extreme summary lengths based on percentiles).
*   **Initial Approach (Mistral 7B):**
    *   Used Mistral's API for initial fine-tuning tests (cost-effective small model).
    *   Developed basic training templates (e.g., `Input: {Claims} Output: {Summary}`).
    *   Identified limitations: model struggled with variable summary lengths. Improved templates by including target paragraph count.
*   **Automated Evaluation Challenges:**
    *   Implemented classical NLP metrics (BLEU, ROUGE) and embedding similarity (cosine similarity) between generated and ground truth summaries using a small SQLite testbed (10 patents).
    *   Explored LLM-as-a-Judge approaches (inspired by MT-Bench, using Prometheus 2 methodology with GPT-4).
    *   **Conclusion:** Automated metrics failed to capture nuances of patent summary quality (technical accuracy, style, hallucination). LLM judges were promising but not yet mature enough for reliable, nuanced evaluation of complex patent text. Manual evaluation remained necessary.
*   **Style Parameter Analysis:**
    *   Manually analyzed numerous patent summaries to identify key stylistic variations.
    *   Defined parameters: Presence/type of intro/conclusion, semantic similarity between claims/summary sentences (`body_similarity`), use of specific connector phrases (`connectors_paragraphs`), ratio of summary length to claims length (`ratio_claim_prompt`).
*   **Automatic Style Annotation Pipeline:**
    *   Developed methods to automatically label summaries with these style parameters:
        *   Intro/Conclusion: Used embedding similarity profiles (cosine similarity of each summary sentence to the claims text) to detect dissimilar starting/ending sentences. LLM-based extraction proved unreliable.
        *   `body_similarity` & `connectors_paragraphs`: Calculated using embeddings and specific LLM prompts.
        *   `ratio_claim_prompt`: Simple length division.
    *   Applied K-means clustering to convert continuous parameters (like `body_similarity`, `ratio_claim_prompt`) into categorical phrases for LLM instructions (e.g., "The body should contain only the main ideas of the claims").
*   **Dataset Creation & Fine-tuning Iterations:**
    *   Created an annotated dataset of ~1000 summaries with style parameters.
    *   Performed multiple fine-tuning iterations using Google's Gemini 1.5 Flash API (free beta access) and OpenAI's GPT-4o mini via Azure ML.
    *   Adjusted fine-tuning hyperparameters (learning rate, batch size, epochs) and refined the dataset based on results. Analyzed loss curves for overfitting/underfitting.
    *   **Key Finding:** Best results obtained using a refined dataset (~1/3 of initial size) focusing on specific, common styles (e.g., `ratio_claim` near 1, high `body_similarity`) and removing sentences too dissimilar from claims (potential added info/hallucinations). This significantly reduced model hallucinations.

    ![Example Loss Curves during LLM Fine-tuning](/images/deepip-internship/slide5_loss_curves.png) # Placeholder - Using Slide 5 graphs from initial slides

**3. Advanced Instruction Design (Adopted Approach):**

*   Implemented techniques to guide pre-trained LLMs (primarily OpenAI models via Azure) for style-controlled summary generation.
*   **Techniques Applied/Evaluated:**
    *   **Chain of Thought (CoT):** Adopted a two-step process inspired by fine-tuning findings:
        1.  LLM first populates a "style template" based on desired parameters (or inferred from an example).
        2.  LLM then generates the summary using the claims *and* the completed style template as instructions.
    *   **Few-Shot Learning:** Used a *single, carefully crafted one-shot example* in the prompt, demonstrating how to apply the style parameters. Adding more examples reduced performance.
    *   **DSPy Framework:** Evaluated DSPy for structured interaction and prompt optimization. Found it promising for modularity but less performant than manually iterated CoT/Few-Shot for this specific task at the time. Optimization gain wasn't deemed sufficient to justify integration for this mature feature.
*   **Focus:** Selecting effective instruction strategies, crafting precise instructions incorporating style parameters, and using the CoT structure.

    ![Instruction Design Flow (CoT) for Style-Controlled Summary Generation](/images/deepip-internship/slide6_prompt_engineering_flow.png) # Placeholder - Using Slide 6 diagram from initial slides (Represents Instruction Design flow now)

### Integrating Style Transfer and Code Refactoring

A significant effort involved integrating the developed style transfer capabilities (allowing users to specify style via instructions or examples) into the main DeepIP-AI codebase.

*   **Motivation:** The existing architecture needed modification to handle the passing and application of style information (parameters, examples) for *every* generation task, not just summaries.
*   **Architecture:** Leveraged the existing **Onion Architecture**.
    *   **Domain Layer:** Core business logic and entities. LLM calls reside here (`AiWriters`). Changes were needed to make `AiWriters` accept and utilize style information.
    *   **Use Case Layer:** Application-specific logic, mapping presentation layer requests to domain actions. Modified to handle style data transfer.
    *   **Presentation Layer:** Interacts with the backend/frontend. Modified to accept style inputs.
*   **Refactoring:**
    *   **Decoupling Style:** The old `StyleEngine` class mixed style definition and application. This was refactored:
        *   `AiWriterStyle` (Dataclass): Created to hold style information (voice mode, free prompting text, example text, banned words). Passed through layers.
        *   `StyleProcessor`: Class responsible for applying style logic (e.g., formatting prompts based on `AiWriterStyle`). Instantiated within `AiWriters`.
        *   `BannedWordsProcessor`: Separated logic for handling banned words (previously part of `BannedWordsMixin`).
    *   **Class Simplification:** Removed `BaseStyleEngine` (unnecessary abstraction) and merged `BannedWordsMixin` functionality into dedicated processors.
    *   Modified all `AiWriter` classes (e.g., `SummarAiWriter`, `ClaimsAiWriter`) and their base class (`BaseAiWriter`) to accept and utilize `StyleProcessor`.
*   **Impact:** Resulted in a more modular, maintainable codebase where style concerns are properly separated and integrated. Involved modifying hundreds of lines of code across multiple classes and updating associated tests (unit, integration, E2E).

**Key Technologies:**

*   **Core Libraries:** Python, asyncio, requests, loguru, pytest, Pydantic, Typer
*   **NLP/ML:** Large Language Models (LLMs: OpenAI API/Azure, Mistral API, Google Gemini API), BERT (for embedding concepts/POCs), Transformers, PyTorch
*   **Embeddings:** OpenAI API (`text-embedding-ada-002`), FAISS, Scikit-learn (for K-means), Cosine Similarity
*   **LLM Specialization:** Fine-tuning (Mistral Platform, Google AI Studio/API, Azure ML), **Instruction Design / In-Context Learning** (Chain of Thought, Few-Shot), DSPy (evaluation)
*   **Data Handling:** Pandas, Parquet, XML parsing (standard library), BeautifulSoup (for scraping POCs), SQLite (for evaluation data)
*   **Development & Ops:** Git (GitLab, GitHub), Docker, Conda, Mac/iTerm, VS Code, Notion, Linear, Slack
*   **External APIs:** EPO OPS (Patent Search API)
*   **Architecture:** Onion Architecture