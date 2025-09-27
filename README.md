# SQuAD MCQ Pipeline

This project converts Stanford Question Answering Dataset (SQuAD) questions into Multiple Choice Questions (MCQs) and evaluates them using an iterative debate pipeline between two LLMs.

## Overview

The pipeline consists of three main components:

1. **SQuAD to MCQ Converter** (`squad_to_mcq_converter.py`) - Converts SQuAD dataset questions into MCQ format with plausible distractors
2. **Debate Pipeline** (`pipeline.py`) - Runs iterative debate between two LLMs to answer MCQ questions  
3. **Runner Script** (`run_squad_pipeline.py`) - Orchestrates the complete workflow

## Requirements

- Python 3.7+
- Required packages: `openai`, `datasets`, `langsmith`, `python-dotenv`
- Together AI API key (set in `.env` file as `TOGETHER_API_KEY`)

## Quick Start

### Option 1: Run Everything at Once (Recommended)

```bash
python run_squad_pipeline.py
```

This single command will:
1. Convert SQuAD questions to MCQ format
2. Run the debate pipeline on the converted questions
3. Generate evaluation results

### Option 2: Run Steps Individually

```bash
# Step 1: Convert SQuAD to MCQ format
python squad_to_mcq_converter.py

# Step 2: Run the debate pipeline
python pipeline.py
```

## Configuration

### SQuAD to MCQ Converter Settings

Edit `squad_to_mcq_converter.py` to modify:

```python
DATASET_NAME = "squad"      # or "squad_v2" for SQuAD 2.0
SPLIT = "validation"        # Dataset split to use
NUM_QUESTIONS = 50          # Number of questions to convert
OUTPUT_FILE = "squad_mcq_questions.json"
```

### Pipeline Settings

Edit `pipeline.py` to modify:

```python
USE_SQUAD_MCQ = True                    # Use SQuAD MCQ vs MMLU
SQUAD_MCQ_FILE = "squad_mcq_questions.json"
NUM_QUESTIONS_TO_PROCESS = 50          # Max questions to process
VERBOSE = False                         # Detailed output
```

### LLM Model Settings

The pipeline uses two models for debate:

```python
LLM1_MODEL = "meta-llama/Llama-3-8b-chat-hf"
LLM2_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MCQ_GENERATION_MODEL = "meta-llama/Llama-3-8b-chat-hf"
```

## Output Files

The pipeline generates several output files:

1. **`squad_mcq_questions.json`** - Converted MCQ questions
2. **`squad_mcq_debate_loop_eval_results_final.json`** - Detailed results with full pipeline output
3. **`squad_mcq_debate_loop_eval_summary_final.csv`** - Summary results in CSV format

## How It Works

### MCQ Generation Process

1. **Load SQuAD Dataset**: Loads questions with context passages and correct answers
2. **Generate Distractors**: Uses an LLM to create plausible but incorrect answer options
3. **Format as MCQ**: Creates multiple choice questions with options A, B, C, D
4. **Random Positioning**: Randomly places the correct answer among the options

### Debate Pipeline Process

1. **Initial Generation**: Both LLMs independently answer each question
2. **Convergence Check**: If both LLMs agree, use that answer
3. **Debate Rounds**: If they disagree, each LLM critiques the other's answer and revises
4. **Iterative Process**: Continue until convergence or maximum rounds reached
5. **Evaluation**: Compare final answers against correct answers

### Example MCQ Question Format

```json
{
  "id": "squad_mcq_1",
  "question": "Context: The Amazon rainforest covers much of the Amazon Basin of South America...\n\nQuestion: What percentage of the Earth's oxygen is produced by the Amazon rainforest?",
  "options": {
    "A": "20%",
    "B": "15%", 
    "C": "25%",
    "D": "10%"
  },
  "correct_answer_letter": "A",
  "original_squad_data": {
    "context": "...",
    "question": "...",
    "answer": "20%",
    "id": "squad_original_id"
  }
}
```

## Evaluation Metrics

The pipeline provides comprehensive evaluation metrics:

- **LLM1 Initial Accuracy**: Performance of first LLM before debate  
- **Final Pipeline Accuracy**: Performance after debate process
- **Improvement Analysis**: Statistical significance of improvements
- **Agreement/Disagreement Rates**: How often LLMs initially agree
- **Convergence Success**: How often debate resolves to correct answers

## Environment Setup

Create a `.env` file with your API keys:

```env
TOGETHER_API_KEY=your_together_ai_api_key_here
LANGCHAIN_PROJECT=SQuAD_MCQ_Debate_Pipeline
```

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `TOGETHER_API_KEY` is set in `.env`
2. **MCQ File Not Found**: Run the converter first or check file permissions
3. **Memory Issues**: Reduce `NUM_QUESTIONS` if running out of memory
4. **Rate Limiting**: Add delays between API calls if hitting rate limits

### Fallback Behavior

If SQuAD MCQ questions aren't available, the pipeline automatically falls back to the original MMLU dataset.

## Customization

### Adding New Datasets

To use other datasets:

1. Modify the converter to load your dataset
2. Ensure questions follow the expected format
3. Update the pipeline configuration

### Changing LLM Models

Update model names in the configuration sections to use different models available through Together AI.

### Adjusting MCQ Generation

Modify the `generate_mcq_options()` function in the converter to:
- Change the number of distractors
- Adjust distractor generation prompts
- Implement different distractor strategies

## License

[Add your license information here]