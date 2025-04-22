---
title: "Data Processing Platform Development at Lingua Custodia"
excerpt: "Developed core Python microservice components for the Datomatic data platform. Created NLP tools, a web scraping framework, and a client library, enhancing financial translation data pipelines"
collection: portfolio
header:
  teaser: "/images/lingua-custodia/datomatic_architecture.png" # Using Figure 12 from the report
# permalink: /portfolio/lingua-custodia-internship/
---

The focus of this project was on developing "Datomatic," a new internal platform for cleaning and processing data to train translation engines, migrating from manual scripts to an automated microservice architecture.

### Datomatic Platform: Architecture and Core Services

Contributed significantly to the design and implementation of the Datomatic platform, built on a microservice architecture.

*   **Microservice Implementation:**
    *   Developed several Python-based microservices, each containerized using **Docker**.
    *   Utilized **ActiveMQ** as the message broker for asynchronous inter-service communication via the **STOMP** protocol.
    *   Integrated services with **EdgeDB**, a modern graph-relational database (used before its production release), for storing processed data (phrases, paragraphs, metadata).
    *   Managed the development environment using **docker-compose** to orchestrate the Python services, ActiveMQ, and EdgeDB containers within a shared network.

    ![Diagram of the Datomatic Architecture showing microservices, ActiveMQ, and EdgeDB](/images/lingua-custodia/datomatic_architecture.png)

*   **Data Cleaning Services:**
    *   **`FixLigatures`:** Developed a service to detect and correct broken typographic ligatures (e.g., "fi" becoming "f i") resulting from PDF-to-text conversions. Used the **`pyahocorasick`** library for efficient multi-pattern string matching based on the Aho-Corasick algorithm. Leveraged Python's `re` module for case-preserving replacements.
    *   **`FilterPorno`:** Created a service to filter out sentences containing excessive vulgar language, preventing contamination of the financial translation models. Implemented a simple scoring system based on predefined word lists.
    *   **Development Practices:** Applied **Test-Driven Development (TDD)** using **pytest** for both unit and integration tests (simulating EdgeDB and ActiveMQ interactions). Employed **Object-Oriented Programming (OOP)** principles to create modular and reusable code, decoupling database/messaging logic from core processing.

*   **Inter-Service Communication Enhancements:**
    *   Addressed limitations in the `stomp.py` library by enabling a single service instance to listen to multiple ActiveMQ destinations simultaneously (e.g., a task queue and an event topic for database updates).
    *   Implemented robust message identification using the `JMSCorrelationID` header and unique UUIDs to track request-response pairs across asynchronous operations, ensuring correct batch processing.
    *   Adapted services to a new **`Message` object architecture** (using Pydantic) designed by the supervisor, enabling complex task chaining (e.g., FixLigatures -> FilterPorno -> Align) via a single initial message containing a `ChainTask` definition.

### NLP Tools: Phrase Alignment and LASER Integration

Developed tools specifically for preparing parallel text data needed for translation model training.

*   **Phrase Alignment Comparison & Implementation:**
    *   Compared existing alignment methods (dictionary-based like AlignFactory/Hunalign, translation-based like BlueAlign) with modern embedding-based approaches.
    *   Evaluated various embedding models (Cohere, OpenAI, Google LaBSE, Facebook LASER) using cosine similarity and margin-based scoring (distance, ratio, absolute).
    *   Implemented the chosen method (LASER) for integration into Datomatic due to its strong performance and language-agnostic nature.

*   **`easylaser` Python Package:**
    *   Created and published ([**PyPI**](https://pypi.org/project/easylaser/)) a user-friendly Python package ([easylaser](https://gitlab.com/linguacustodia/easylaser)) to wrap Facebook's LASER v2/v3 embeddings.
    *   Simplified usage by handling model/dependency downloads automatically, accepting Python lists as input/output (instead of files), and removing the need for external Perl scripts.
    *   Implemented **multi-GPU support** using Python's `multiprocessing` module (`MultiGpuEncoder` class) to significantly accelerate embedding generation.
    *   Integrated phrase alignment logic directly into the package.

### Data Acquisition: Web Scraping Framework

Built a framework to automate the collection of bilingual data from online sources.

*   **Framework Design:**
    *   Developed an extensible **OOP-based scraping framework**. Core components include an abstract `Scraper` class (requiring site-specific `_get_urls` and `_get_content` implementations), `Collection` (managing `Insertable` data objects), and `Insertable` (abstract class mapping to EdgeDB tables).
    *   Used **BeautifulSoup** and **requests** for HTML parsing and fetching.
    *   Implemented a `ScraperManager` using Python's `multiprocessing` to run multiple scrapers concurrently and periodically (e.g., daily).

*   **Integration:**
    *   Automatically inserted scraped data (bilingual sentences, paragraphs, documents) into **EdgeDB**, handling object relationships and generating dynamic, nested insertion queries.
    *   Integrated with a **MinIO** DataLake (deployed via Docker) to store original source files (e.g., PDFs, HTML articles) for reference and potential LLM training, linking them to the processed data in EdgeDB.
    *   Created scrapers for specific bilingual financial news sites (e.g., FT Chinese).

    ![Class Diagram of the ScraperManager and related components](/images/lingua-custodia/scraper_manager_diagram.png)
### Supporting Tools and DevOps

*   **`python-lib-dato` Client Library:**
    *   Co-developed an internal Python library providing a high-level API for users to submit data processing tasks to the Datomatic platform.
    *   Handled sending `Message` objects with task chains to ActiveMQ, receiving results, and managing task progress tracking via EdgeDB (`TaskNameHandler`).
    *   Implemented functionality to process TSV files/folders and manage concurrent requests using **multithreading** (`TaskMessenger` class) to avoid overloading ActiveMQ, including a callback system for handling results efficiently.

    ![Schema of the python-lib-dato interaction with Datomatic](/images/lingua-custodia/python_lib_dato_schema.png)

*   **GitLab CI/CD:**
    *   Created a **GitLab CI/CD pipeline** using YAML to automate the updating of Git submodules within the main Datomatic repository.
    *   Configured a **Git Runner** and used protected GitLab access tokens to handle permissions issues, ensuring seamless integration and repository consistency across different components.

**Key Technologies:**

*   **Core Libraries:** Python (asyncio, multiprocessing, multithreading, Pydantic, pytest, requests, stomp.py, re), Linux/SSH
*   **Data Processing & NLP:** Pandas, BeautifulSoup, pyahocorasick, LASER embeddings, SentencePiece
*   **Infrastructure & Databases:** Docker, docker-compose, ActiveMQ (STOMP), EdgeDB, MinIO (S3 compatible)
*   **Development & Ops:** Git (GitLab), GitLab CI/CD, VS Code (Pylint, Copilot), Trello, Slack