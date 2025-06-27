# Block 1: Imports and Setup
import os
import json
import csv
import re
import uuid
import math # For sigma calculation
from openai import OpenAI
from langsmith import traceable
from dotenv import load_dotenv
from datasets import load_dataset

# Load secrets from our .env file
load_dotenv()

# Configure LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "TogetherAI_MMLU_Final_Eval_Metrics")

# Initialize the OpenAI client to point to Together AI
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("ERROR: TOGETHER_API_KEY not found in environment variables. Please set it in your .env file.")
    exit()

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)

# ---

# Block 2: Worker Functions (Unchanged - logic is the same)
# Block 2: Worker Functions with More Forceful Prompts

LLM1_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
LLM2_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

@traceable
def ask_llm1_to_answer(question_text: str, options_text: str, verbose: bool = False):
    ### --- UPDATED & STRICTER PROMPT --- ###
    system_prompt = (
        "You are an automated agent designed ONLY to take multiple-choice tests. "
        "You MUST select the best possible answer from the given options, even if you believe the question or options could be improved. "
        "Do not refuse to answer. Do not critique the question. Your only task is to choose the best option. "
        "First, provide a short justification for your choice, and then on the SAME LINE, provide the letter choice in parentheses.\n"
        "Your entire response must be a SINGLE LINE formatted exactly as:\n"
        "<SHORT JUSTIFICATION> (<LETTER>)\n\n"
        "For example:\n"
        "The Declaration of Independence was primarily authored by Thomas Jefferson. (A)"
    )
    full_prompt = f"{question_text}\n{options_text}"
    if verbose:
        print(f"  Asking LLM1 ({LLM1_MODEL}) with CoT prompt:\n{full_prompt[:200]}...")
    response = client.chat.completions.create(
        model=LLM1_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content

@traceable
def ask_llm2_to_check(question_text: str, options_text: str, answer_from_llm1: str, verbose: bool = False):
    # This prompt is already strict and does not need changes.
    system_prompt = (
        "You are a verifier. "
        "Respond with **exactly** the single word 'Agree' or 'Disagree' "
        "without punctuation."
    )
    full_prompt = f"{question_text}\n{options_text}"
    if verbose:
        print(f"  Asking LLM2 ({LLM2_MODEL}) to check answer: {answer_from_llm1[:100]}...")
    response = client.chat.completions.create(
        model=LLM2_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Question:\n{full_prompt}\n\nProposed answer:\n{answer_from_llm1}"}]
    )
    return response.choices[0].message.content

@traceable
def ask_llm2_for_feedback(question_text: str, options_text: str, incorrect_answer: str, verbose: bool = False):
    # This prompt is also fine as it's meant to be freeform.
    system_prompt = (
        "You are a teaching assistant. "
        "The answer above is incorrect. Explain the error **briefly** "
        "and point to the correct reasoning (2–3 sentences)."
    )
    full_prompt = f"{question_text}\n{options_text}"
    if verbose:
        print(f"  Asking LLM2 ({LLM2_MODEL}) for feedback on: {incorrect_answer[:100]}...")
    response = client.chat.completions.create(
        model=LLM2_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Question:\n{full_prompt}\n\nIncorrect answer:\n{incorrect_answer}"}]
    )
    return response.choices[0].message.content

@traceable
def ask_llm1_to_retry_with_feedback(question_text: str, options_text: str, feedback: str, verbose: bool = False):
    ### --- UPDATED & STRICTER PROMPT --- ###
    system_prompt = (
        "You previously answered incorrectly. Use the provided feedback to re-evaluate the question and choose the best option. "
        "Do not refuse to answer. Do not critique the question. Your only task is to choose the best option. "
        "First, write a new, short justification based on the feedback, "
        "then on the SAME LINE, provide the new option letter in parentheses.\n"
        "Your entire response must be a SINGLE LINE formatted exactly as:\n"
        "<NEW JUSTIFICATION> (<LETTER>)"
    )
    full_prompt = f"{question_text}\n{options_text}"
    if verbose:
        print(f"  Asking LLM1 ({LLM1_MODEL}) to retry with CoT prompt...")
    response = client.chat.completions.create(
        model=LLM1_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Original question:\n{full_prompt}\n\nFeedback on your mistake:\n{feedback}"}]
    )
    return response.choices[0].message.content

# Block 3: Orchestrator function (Unchanged)
@traceable(run_type="chain")
def mmlu_pipeline(question_data: dict, verbose: bool = False):
    question_text = question_data["question"]
    options_list  = [f"({k}) {v}" for k, v in question_data["options"].items()]
    options_text  = "\n".join(options_list)

    llm1_answer   = ask_llm1_to_answer(question_text, options_text, verbose=verbose)
    verification  = ask_llm2_to_check(question_text, options_text, llm1_answer, verbose=verbose)

    pipeline_result = {"llm1_initial_answer": llm1_answer, "llm2_verification": verification, "llm2_feedback": None, "llm1_corrected_answer": None, "final_answer_for_evaluation": llm1_answer}
    processed_verification = verification.strip().capitalize()

    if verbose:
        print(f"\nLLM1's Initial Answer:\n---\n{llm1_answer}\n---")
        print(f"LLM2's Verification: {processed_verification}")

    if processed_verification == "Agree":
        if verbose: print("\nPipeline finished. Models in agreement.")
    else:
        if verbose:
            if processed_verification == "Disagree": print("\nModels disagree. Starting feedback loop...")
            else: print(f"\nLLM2's Verification (Unexpected): '{verification}'\nWarning: Treating as disagreement for safety.")
        feedback = ask_llm2_for_feedback(question_text, options_text, llm1_answer, verbose=verbose)
        if verbose: print(f"\nFeedback from LLM2:\n---\n{feedback}\n---")
        pipeline_result["llm2_feedback"] = feedback
        if processed_verification != "Disagree": pipeline_result["llm2_verification"] = f"Disagree (Interpreted from: {verification})"
        final_answer = ask_llm1_to_retry_with_feedback(question_text, options_text, feedback, verbose=verbose)
        if verbose: print("\nPipeline finished after correction.")
        pipeline_result["llm1_corrected_answer"] = final_answer
        pipeline_result["final_answer_for_evaluation"] = final_answer

    return pipeline_result

# ---

# Block 4: Letter-extraction helper (Unchanged)
def get_letter_from_output(text_output: str | None):
    if not text_output: return None
    clean_text = text_output.strip()
    m1 = re.search(r'\(([A-Ea-e])\)\s*$', clean_text)
    if m1: return m1.group(1).upper()
    m2 = re.search(r'\(([A-Ea-e])\)', clean_text)
    if m2: return m2.group(1).upper()
    m3 = re.match(r'^\s*([A-Ea-e])\.?\s*$', clean_text)
    if m3: return m3.group(1).upper()
    m4 = re.search(r'is\s+([A-Ea-e])[\.\s,]*$', clean_text, re.IGNORECASE)
    if m4: return m4.group(1).upper()
    print(f"Warning: Could not reliably parse letter from output: '{text_output[:100]}...'")
    return None

# ---

# Block 5: Dataset Loading, Evaluation Loop, Metrics, and Saving Results
if __name__ == "__main__":
    # --- SCRIPT CONFIGURATION ---
    VERBOSE = False 
    MMLU_SUBJECT = "all"  # <<< Use "all" to access all 57 MMLU subjects
    NUM_QUESTIONS_TO_PROCESS = 1000 # Set to your desired number
    SHUFFLE_SEED = 42
    # ---------------------------

    print(f"Starting evaluation for {NUM_QUESTIONS_TO_PROCESS} questions from MMLU subject: '{MMLU_SUBJECT}'...")
    
    hf_dataset = None
    try:
        full_subject_dataset = load_dataset("cais/mmlu", MMLU_SUBJECT, split="test", trust_remote_code=True)
        shuffled_dataset = full_subject_dataset.shuffle(seed=SHUFFLE_SEED)
        
        if len(shuffled_dataset) < NUM_QUESTIONS_TO_PROCESS:
            print(f"Warning: Requested {NUM_QUESTIONS_TO_PROCESS} questions, but the dataset only has {len(shuffled_dataset)}. Processing all available.")
            hf_dataset = shuffled_dataset
        else:
            hf_dataset = shuffled_dataset.select(range(NUM_QUESTIONS_TO_PROCESS))
        print(f"Loaded and shuffled {len(hf_dataset)} questions.")

    except Exception as e:
        print(f"Failed to load dataset from Hugging Face: {e}\nExiting.")
        exit()

    print(f"--- Starting Pipeline Evaluation using Together AI ---")
    
    all_results = []
    option_letters = ["A", "B", "C", "D", "E"] 

    for i, item in enumerate(hf_dataset):
        if not VERBOSE:
            print(f"Processing question {i+1}/{len(hf_dataset)}...", end='\r')
        
        # --- Data Adaptation for MMLU format ---
        question_text = item["question"]
        current_options = {option_letters[idx]: text for idx, text in enumerate(item["choices"]) if idx < len(option_letters)}
        
        if not current_options:
            if VERBOSE: print(f"Skipping item {i+1} due to no choices.")
            continue
        
        correct_answer_index = item["answer"]
        correct_answer_letter = option_letters[correct_answer_index] if 0 <= correct_answer_index < len(current_options) else "N/A"
        # --- End Data Adaptation ---

        question_data_for_pipeline = {"id": f"hf_mmlu_{item.get('subject', 'general')}_{SHUFFLE_SEED}_{i+1}", "question": question_text, "options": current_options, "correct_answer_letter": correct_answer_letter}
        
        if VERBOSE:
            print(f"\n--- Processing Question {i+1}/{len(hf_dataset)} (ID: {question_data_for_pipeline['id']}) ---")
        
        try:
            pipeline_output = mmlu_pipeline(question_data_for_pipeline, verbose=VERBOSE)
        except Exception as e:
            if VERBOSE: print(f"!!!!!!!! ERROR processing pipeline for question ID {question_data_for_pipeline['id']}: {e} !!!!!!!!")
            pipeline_output = {"final_answer_for_evaluation": "ERROR"}
        
        if VERBOSE:
            print("\n===================================")
            print(f"✅ Final Answer for Question ID {question_data_for_pipeline['id']}:\n{pipeline_output['final_answer_for_evaluation']}")
            print(f"   Correct Answer Letter: {correct_answer_letter}")
            print("===================================")
        
        result_summary = {**question_data_for_pipeline, "pipeline_output_details": pipeline_output}
        all_results.append(result_summary)

    # Print a newline to clear the progress indicator and add a separator
    print("\n" + "="*50)
    print("--- Evaluation Loop Finished ---")
    print(f"Processed {len(all_results)} questions.")
    
    # --- Calculate Evaluation Metrics and Sigma ---
    # ... (This logic remains the same) ...
    total_questions_attempted = len(all_results)
    llm1_initial_correct_count, agreement_count, final_pipeline_correct_count = 0, 0, 0
    disagreements_count, corrected_after_disagreement_count, error_in_pipeline_count = 0, 0, 0

    if total_questions_attempted > 0:
        for res in all_results:
            if res['pipeline_output_details'].get('final_answer_for_evaluation') == "ERROR":
                error_in_pipeline_count +=1; continue
            llm1_initial_answer_letter = get_letter_from_output(res['pipeline_output_details']['llm1_initial_answer'])
            final_answer_letter = get_letter_from_output(res['pipeline_output_details']['final_answer_for_evaluation'])
            correct_letter = res['correct_answer_letter']
            if llm1_initial_answer_letter == correct_letter: llm1_initial_correct_count += 1
            verification_text = res['pipeline_output_details']['llm2_verification']
            if "Agree" in verification_text.strip().capitalize(): agreement_count += 1
            elif verification_text != "ERROR":
                disagreements_count += 1
                if res['pipeline_output_details']['llm1_corrected_answer'] and res['pipeline_output_details']['llm1_corrected_answer'] != "ERROR":
                    corrected_answer_letter = get_letter_from_output(res['pipeline_output_details']['llm1_corrected_answer'])
                    if corrected_answer_letter == correct_letter and llm1_initial_answer_letter != correct_letter:
                        corrected_after_disagreement_count += 1
            if final_answer_letter == correct_letter: final_pipeline_correct_count += 1
        
        valid_runs_for_metrics = total_questions_attempted - error_in_pipeline_count
        print("\n--- Final Evaluation Summary ---")
        if error_in_pipeline_count > 0: print(f"Pipeline Errors: {error_in_pipeline_count}/{total_questions_attempted}")
        if valid_runs_for_metrics > 0:
            p1 = llm1_initial_correct_count / valid_runs_for_metrics
            p2 = final_pipeline_correct_count / valid_runs_for_metrics
            sigma1 = math.sqrt(p1 * (1 - p1) / valid_runs_for_metrics) if valid_runs_for_metrics > 0 else 0
            sigma2 = math.sqrt(p2 * (1 - p2) / valid_runs_for_metrics) if valid_runs_for_metrics > 0 else 0
            total_sigma = math.sqrt(sigma1**2 + sigma2**2) if (sigma1 > 0 or sigma2 > 0) else 0
            improvement = p2 - p1
            significance_in_sigma = improvement / total_sigma if total_sigma > 0 else float('inf')
            print(f"LLM1 Initial Accuracy : {p1:.2%} (± {sigma1:.2%})")
            print(f"Final Pipeline Accuracy: {p2:.2%} (± {sigma2:.2%})")
            print("-" * 30)
            print(f"Improvement: {improvement:+.2%}")
            print(f"Significance: {significance_in_sigma:.2f} sigma")
            if abs(significance_in_sigma) < 1.96: print("(Improvement is NOT statistically significant at 95% confidence level)")
            else: print("(Improvement IS statistically significant at 95% confidence level)")
        else: print("No valid runs completed to calculate metrics.")
    else: print("No questions processed.")

    # --- Storing results to JSON and CSV files ---
    json_output_filename = f"mmlu_all_subjects_eval_results.json"
    csv_output_filename = f"mmlu_all_subjects_eval_summary.csv"
    # ... (The rest of the file saving logic remains the same) ...