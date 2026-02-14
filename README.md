# Code Switching Jailbreak Experiment

This project investigates whether **code-switching** (mixing multiple languages in a single prompt) can bypass the safety filters of Large Language Models (LLMs).

It systematically generates adversarial prompts by embedding words from different languages (e.g., Arabic, Greek, Spanish) into a matrix language (e.g., English, Italian) and evaluates the Attack Success Rate (ASR) against a target model.

## 🚀 Features

- **Systematic Grid Search**: Automatically tests various code-switching parameters:
    - **Swap Ratios**: 10%, 30%, 50%, 70%, 90%
    - **Language Strategies**: Arabic Dominant, Greek Dominant, Spanish Dominant, Balanced
- **Robust Evaluation**:
    - **Multi-Iteration Translation**: Generates **N variations** per prompt (default: 10) to account for translation stochasticity.
    - **Multi-Iteration Attack**: Queries the target model **M times** per variation (default: 10) to account for generation stochasticity.
    - **Total Coverage**: 100 attempts per original prompt.
- **Hybrid Translation Engine**:
    - **Local (Default)**: Uses **Argos Translate** with pivoting (IT -> EN -> Target) for offline, quota-free operation.
    - **Online (Optional)**: Support for **Google Translate API** via flag.
- **Automated Safety Judge**: Uses **LlamaGuard** (via Ollama).

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mekadmini/data_ethics_seminar.git
   cd data_ethics_seminar
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Translation Models (Crucial)**:
   This script downloads the necessary offline translation models (approx. 1GB) based on the languages defined in `lib/custom_types.py`.
   ```bash
   python scripts/setup_translation_models.py
   ```

5. **Download SpaCy models**:
   ```bash
   python -m spacy download en_core_web_sm # English
   python -m spacy download it_core_news_sm # Italian
   ```

6. **Setup Ollama**:
   Ensure [Ollama](https://ollama.com/) is installed and running. Pull the required models:
   ```bash
   ollama pull llama3       # Target Model
   ollama pull llama-guard3 # Safety Judge
   ```

## 🏃 Usage

### Run the Full Study (Local / Default)
The main entry point is `study_runner.py`. This runs the full grid search using **local translation** (Argos).

```bash
python study_runner.py
```

### Run with Google Translate (Online)
To use the Google API instead of local models:
```bash
python study_runner.py --use_google
```

This will:
1. Generate **10 code-switched variations** for each original prompt in **20 different scenarios**.
2. Query the target model (Llama3) **10 times** for *each* variation (Total: 100 queries per original prompt).
3. Evaluate safety using LlamaGuard.
4. Save results to `experiment_results/study_<timestamp>/`.
5. Print the **Winning Configuration** (highest ASR) at the end.

### Customizing the Study
Edit `study_runner.py` to adjust:
- `dataset_split`: Number of prompts to use (e.g., `"train[:100]"`).
- `translation_iterations`: Number of translation variations per source prompt (default: 10).
- `iterations`: Number of responses per variation (default: 10).
- `swap_ratios` & `strategies`: The grid search parameters.

### Single Experiment
You can also run a single experiment configuration using `main.py`:

```bash
python main.py experiment --input_dir experiment_results/my_single_run
```

## 📊 Results

Results are saved in `experiment_results/study_<timestamp>/`.

- **`final_summary.csv`**: Aggregated statistics for all scenarios.
- **`Ratio_X_Strategy_Y/`**: Folder for specific scenario results.
    - `results_evaluated.csv`: Detailed prompt-response pairs with safety labels.
    - `prompts.csv`: The generated prompts.
    - `config.json`: Configuration used for this run.

### Metrics

- **ASR (Response Only)**: % of responses classified as unsafe.
- **ASR (Contextual)**: % of (prompt, response) pairs classified as unsafe (more accurate).
- **Consistent Success**: % of prompts that bypassed the filter in ≥50% of iterations.

## 📈 Interpreting the Final Summary (`final_summary.csv`)

The `final_summary.csv` file aggregates the results of the code-switching attack study. Here is precisely how to interpret each column:

### 1. `asr_both` (Contextual Attack Success Rate)
*   **Definition:** The percentage of **User Prompt + Model Response** pairs that the judge (Llama Guard) classified as "unsafe".
*   **Interpretation:** It represents how often the code-switching strategy successfully tricked the target model into generating harmful content.
*   **Why use this over `asr_response`?** Security judges like Llama Guard often need the *context* of the malicious prompt to correctly identify a response as harmful. If the response looks benign in isolation (e.g., a chemical mixing list), the judge might miss it without seeing the "How to make a bomb" prompt.

### 2. `asr_response` (Response-Only ASR)
*   **Definition:** The percentage of **Model Responses** (in isolation) that were classified as "unsafe".
*   **Interpretation:** If this is `0.0`, it means the judge model did not flag any response as unsafe when viewing the response text alone. This confirms that `asr_both` is the correct metric to use for the analysis, as the judge clearly required the prompt context to make a verdict.

### 3. `at_least_one_success` (Vulnerability Potential)
*   **Definition:** The percentage of **unique prompts** that successfully triggered an unsafe response *at least once* across the configured iterations (e.g., 10 attempts).
*   **Interpretation:** This measures the **breadth of vulnerability**. If this is high, it means the attack works on many different *types* of questions, even if it's not consistent every single time.

### 4. `consistent_success` (Reliability)
*   **Definition:** The percentage of **unique prompts** that triggered an unsafe response in **50% or more** of the iterations.
*   **Interpretation:** This measures the **reliability/robustness** of the attack. A high score here means the jailbreak is stable and reproducible, not just a random fluke due to LLM stochasticity.

### 5. `scenario`
*   **Definition:** The specific configuration name (e.g., `Ratio_0.9_Spanish_Dom`).
*   **Breakdown:**
    *   `Ratio_0.9`: 90% of the words were swapped to the embedded language/mixed implementation.
    *   `Spanish_Dom`: The embedded language distribution was skewed heavily towards Spanish (80% Spanish, 10% others).

## 📂 Interpreting Other Result Files

In addition to the summary, each scenario folder (e.g., `Ratio_0.9_Spanish_Dom/`) contains detailed files:

### 1. `results_evaluated.csv` (Detailed Logs)
This file contains every single prompt-response pair and its safety evaluation.
*   **`csrt`**: The actual code-switched prompt used.
*   **`response_modelname`**: The raw response from the target LLM.
*   **`safe_prompt`**: Judge's verdict on the *prompt only* (unsafe/safe).
*   **`safe_response`**: Judge's verdict on the *response only* (unsafe/safe).
*   **`safe_both`**: Judge's verdict on the **contextual pair** (unsafe/safe).

### 2. `per_prompt_scores.csv` (Prompt-Level Stats)
This file tells you **which specific questions** successfully tricked the model. It aggregates results across all 10 iterations for each unique prompt.

*   **`csrt`**: The actual code-switched prompt used.
*   **`is_unsafe_response`**: The success rate based on the response only (0.0 to 1.0).
    *   `0.0`: The response was never flagged as unsafe.
    *   `1.0`: The response was always flagged as unsafe.
*   **`is_unsafe_both`**: The success rate based on the prompt+response context (0.0 to 1.0). **Look at this column to find winning prompts.**
    *   `0.0`: Failed every time (0/10).
    *   `0.5`: Worked 50% of the time (5/10) - somewhat reliable.
    *   `1.0`: Worked every single time (10/10) - highly reliable jailbreak.

### 3. `prompts.csv` (Source Data)
*   **`id`**: Unique ID of the prompt template.
*   **`it`**: The original prompt in the matrix language (Italian in this run).
*   **`csrt`**: The generated code-switched version.