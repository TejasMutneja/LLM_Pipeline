# Block 1: Imports and Setup
import os
import json
import csv
import re
import uuid
import math
from openai import OpenAI
from langsmith import traceable
from dotenv import load_dotenv
from datasets import load_dataset

# Load secrets from our .env file
load_dotenv()

# Configure LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "TogetherAI_Debate_Parser_Fix_Eval")

# Initialize the OpenAI client to point to Together AI
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("ERROR: TOGETHER_API_KEY not found in environment variables.")
    exit()

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)

# ---

# Block 2: Worker Functions (Updated to accept verbose flag)
LLM1_MODEL = "meta-llama/Llama-3-8b-chat-hf"
LLM2_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

@traceable
def generate_answer(model_name: str, question_text: str, options_text: str, verbose: bool = False):
    system_prompt = """
    You are an expert exam-taker. Your task is to select the best possible answer from the given options.

    Follow this EXACT format for your response:
    1. Write your step-by-step reasoning
    2. End your response with EXACTLY:
    Final Answer: (X)

    where X is a single letter A, B, C, D, or E.
    Example ending: Final Answer: (A)

    Your response MUST end with the Final Answer line in exactly this format."""
    full_prompt = f"{question_text}\n{options_text}"
    if verbose: print(f"  Asking {model_name} for initial answer...")
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content

@traceable
def critique_and_revise_answer(model_name: str, question_text: str, options_text: str, peer_answer: str, verbose: bool = False):
    system_prompt = (
        "You are an expert reviewer evaluating a peer's answer. "
        "First, write out your critical evaluation of the peer's reasoning. "
        "Then, write out your own step-by-step reasoning to arrive at the best answer. "
        "Finally, conclude your entire response with a new line containing ONLY the phrase:\n"
        "Final Answer: (<LETTER>)\n"
        "where <LETTER> is your chosen option."
    )
    full_prompt = f"{question_text}\n{options_text}"
    user_prompt = (
        f"The Question:\n{full_prompt}\n\n"
        f"A peer AI provided this answer:\n{peer_answer}\n\n"
        "Critically evaluate the peer's answer and provide your own definitive, final answer."
    )
    if verbose: print(f"  Asking {model_name} to critique and revise...")
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    )
    return response.choices[0].message.content

# ---

# Block 3: Orchestrator function (Updated to accept verbose flag)
@traceable(run_type="chain")
def mmlu_pipeline(question_data: dict, verbose: bool = False):
    MAX_ROUNDS = 10
    question_text = question_data["question"]
    options_text = "\n".join([f"({k}) {v}" for k, v in question_data["options"].items()])

    if verbose: print("\n--- Round 0: Initial Parallel Generation ---")
    llm1_current_answer = generate_answer(LLM1_MODEL, question_text, options_text, verbose)
    llm2_current_answer = generate_answer(LLM2_MODEL, question_text, options_text, verbose)

    discussion_history = [{"round": 0, "llm1_answer": llm1_current_answer, "llm2_answer": llm2_current_answer}]
    if verbose:
        print(f"LLM1 Initial: {llm1_current_answer[:100]}...")
        print(f"LLM2 Initial: {llm2_current_answer[:100]}...")

    for round_num in range(MAX_ROUNDS):
        if verbose: print(f"\n--- Round {round_num + 1}: Convergence Check & Debate ---")
        
        llm1_letter = get_letter_from_output(llm1_current_answer)
        llm2_letter = get_letter_from_output(llm2_current_answer)

        if llm1_letter is not None and llm1_letter == llm2_letter:
            if verbose: print(f"Convergence reached in Round {round_num + 1}. Final Answer Letter: ({llm1_letter})")
            break
        
        if verbose: print(f"No convergence (LLM1: {llm1_letter}, LLM2: {llm2_letter}). Proceeding with debate.")

        llm1_previous_answer, llm2_previous_answer = llm1_current_answer, llm2_current_answer
        llm1_current_answer = critique_and_revise_answer(LLM1_MODEL, question_text, options_text, llm2_previous_answer, verbose)
        llm2_current_answer = critique_and_revise_answer(LLM2_MODEL, question_text, options_text, llm1_previous_answer, verbose)

        discussion_history.append({"round": round_num + 1, "llm1_answer": llm1_current_answer, "llm2_answer": llm2_current_answer})
        if verbose:
            print(f"LLM1 Revised Answer: {llm1_current_answer[:100]}...")
            print(f"LLM2 Revised Answer: {llm2_current_answer[:100]}...")
    else:
        if verbose: print(f"\nMax rounds ({MAX_ROUNDS}) reached. Pipeline finished without convergence.")

    return {"final_answer_for_evaluation": llm1_current_answer, "discussion_history": discussion_history}

# ---

# Block 4: Letter-extraction helper (Unchanged)
def get_letter_from_output(text_output: str | None):
    if not text_output:
        return None
    
    # Convert to uppercase and strip whitespace
    text_output = text_output.upper().strip()
    
    # List of regex patterns to try in order of preference
    patterns = [
        r'FINAL ANSWER:\s*\(([A-E])\)',  # Standard format: Final Answer: (A)
        r'FINAL ANSWER:\s*([A-E])\)?',    # Variation without parentheses
        r'\(([A-E])\)\s*$',               # Just (A) at the end
        r'ANSWER\s*(?:IS|:)?\s*\(([A-E])\)', # Answer is (A)
        r'(?:CHOOSE|SELECT|PICK)\s*\(([A-E])\)', # Choose/Select/Pick (A)
        r'(?:^|\s|\n)([A-E])\)(?:\s|$)',  # A) with boundary
        r'\(([A-E])\)'                     # Last resort: any (A) in the text
    ]
    
    # Try each pattern in order
    for pattern in patterns:
        match = re.search(pattern, text_output)
        if match:
            return match.group(1).upper()
    
    # If we still haven't found a match, try to find any single letter A-E
    single_letters = re.findall(r'(?:^|\s)([A-E])(?:\s|$)', text_output)
    if single_letters:
        # Return the last single letter found
        return single_letters[-1]
    
    if "ERROR" in text_output:
        return None
        
    print(f"Warning: Could not parse letter from output: '{text_output[-200:]}'")
    return None
# ---

# Block 5: Dataset Loading, Evaluation Loop, Metrics, and Saving Results

if __name__ == "__main__":
    # --- SCRIPT CONFIGURATION ---
    VERBOSE = False 
    DATASET_NAME = "cais/mmlu"
    DATASET_CONFIG = "all"
    NUM_QUESTIONS_TO_PROCESS = 1000
    SHUFFLE_SEED = 42
    # ---------------------------

    print(f"Starting evaluation for {NUM_QUESTIONS_TO_PROCESS} questions from '{DATASET_NAME}' ({DATASET_CONFIG} subset)...")
    
    hf_dataset = None
    try:
        full_subject_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test", trust_remote_code=True)
        shuffled_dataset = full_subject_dataset.shuffle(seed=SHUFFLE_SEED)
        if len(shuffled_dataset) < NUM_QUESTIONS_TO_PROCESS: hf_dataset = shuffled_dataset
        else: hf_dataset = shuffled_dataset.select(range(NUM_QUESTIONS_TO_PROCESS))
        print(f"Loaded and shuffled {len(hf_dataset)} questions.")
    except Exception as e:
        print(f"Failed to load dataset from Hugging Face: {e}\nExiting.")
        exit()

    print(f"--- Starting Iterative Debate Pipeline Evaluation using Together AI ---")
    
    all_results, option_letters = [], ["A", "B", "C", "D", "E"]

    for i, item in enumerate(hf_dataset):
        if not VERBOSE:
            print(f"Processing question {i+1}/{len(hf_dataset)}...", end='\r')
        
        question_text = item["question"]
        current_options = {option_letters[idx]: text for idx, text in enumerate(item["choices"]) if idx < len(option_letters)}
        if not current_options: continue
        
        correct_answer_index = item["answer"]
        correct_answer_letter = option_letters[correct_answer_index] if 0 <= correct_answer_index < len(current_options) else "N/A"

        question_data_for_pipeline = {"id": f"hf_{DATASET_CONFIG}_{SHUFFLE_SEED}_{i+1}", "question": question_text, "options": current_options, "correct_answer_letter": correct_answer_letter}
        
        if VERBOSE: print(f"\n--- Processing Question {i+1}/{len(hf_dataset)} (ID: {question_data_for_pipeline['id']}) ---")
        
        try:
            pipeline_output = mmlu_pipeline(question_data_for_pipeline, verbose=VERBOSE)
        except Exception as e:
            if VERBOSE: print(f"!!!!!!!! ERROR processing pipeline for question ID {question_data_for_pipeline['id']}: {e} !!!!!!!!")
            pipeline_output = {"final_answer_for_evaluation": "ERROR"}
        
        result_summary = {**question_data_for_pipeline, "pipeline_output_details": pipeline_output}
        all_results.append(result_summary)

    print("\n" + "="*50)
    print("--- Evaluation Loop Finished ---")
    print(f"Processed {len(all_results)} questions.")

    ### --- METRICS BLOCK UPDATED --- ###
    total_questions_attempted = len(all_results)
    llm1_initial_correct_count, llm2_initial_correct_count, final_pipeline_correct_count, error_in_pipeline_count = 0, 0, 0, 0
    initial_agreement_count = 0
    initial_disagreement_count = 0
    convergence_to_correct_count = 0

    if total_questions_attempted > 0:
        for res in all_results:
            if res['pipeline_output_details'].get('final_answer_for_evaluation') == "ERROR":
                error_in_pipeline_count +=1; continue
            
            initial_answers = res['pipeline_output_details'].get('discussion_history', [{}])[0]
            llm1_initial_letter = get_letter_from_output(initial_answers.get('llm1_answer'))
            llm2_initial_letter = get_letter_from_output(initial_answers.get('llm2_answer'))
            final_answer_letter = get_letter_from_output(res['pipeline_output_details']['final_answer_for_evaluation'])
            correct_letter = res['correct_answer_letter']
            
            if llm1_initial_letter == correct_letter: llm1_initial_correct_count += 1
            if llm2_initial_letter == correct_letter: llm2_initial_correct_count += 1
            if final_answer_letter == correct_letter: final_pipeline_correct_count += 1

            if llm1_initial_letter is not None and llm1_initial_letter == llm2_initial_letter:
                initial_agreement_count += 1
            else:
                initial_disagreement_count += 1
                # A "successful correction" means an initial disagreement resolved to the correct answer.
                if final_answer_letter == correct_letter:
                    convergence_to_correct_count += 1
        
        valid_runs_for_metrics = total_questions_attempted - error_in_pipeline_count
        print("\n--- Final Evaluation Summary (Iterative Debate Pipeline) ---")
        if error_in_pipeline_count > 0: print(f"Pipeline Errors: {error_in_pipeline_count}/{total_questions_attempted}")
        
        if valid_runs_for_metrics > 0:
            # Calculate base accuracies
            p_llm1 = llm1_initial_correct_count / valid_runs_for_metrics
            p_final = final_pipeline_correct_count / valid_runs_for_metrics
            
            # Calculate sigma (standard error) for initial vs final
            sigma1 = math.sqrt(p_llm1 * (1 - p_llm1) / valid_runs_for_metrics) if valid_runs_for_metrics > 0 else 0
            sigma_final = math.sqrt(p_final * (1 - p_final) / valid_runs_for_metrics) if valid_runs_for_metrics > 0 else 0
            
            # Calculate total sigma and significance of the improvement over LLM1
            total_sigma_vs_llm1 = math.sqrt(sigma1**2 + sigma_final**2) if (sigma1 > 0 or sigma_final > 0) else 0
            improvement_vs_llm1 = p_final - p_llm1
            significance_vs_llm1 = improvement_vs_llm1 / total_sigma_vs_llm1 if total_sigma_vs_llm1 > 0 else float('inf')

            print(f"LLM1 Initial Accuracy         : {p_llm1:.2%} (± {sigma1:.2%})")
            print(f"Final Pipeline Accuracy       : {p_final:.2%} (± {sigma_final:.2%})")
            print("-" * 40)
            print(f"Improvement over LLM1         : {improvement_vs_llm1:+.2%}")
            print(f"Significance vs LLM1          : {significance_vs_llm1:.2f} sigma")
            if abs(significance_vs_llm1) < 1.96: print("(Improvement over LLM1 is NOT statistically significant)")
            else: print("(Improvement over LLM1 IS statistically significant)")
            
            # Add other relevant debate metrics
            print("\n--- Debate Process Metrics ---")
            print(f"Initial Agreement Rate        : {initial_agreement_count / valid_runs_for_metrics:.2%}")
            if initial_disagreement_count > 0:
                print(f"Convergence to Correct Rate   : {convergence_to_correct_count / initial_disagreement_count:.2%} ({convergence_to_correct_count}/{initial_disagreement_count})")
            else:
                print("No initial disagreements occurred to measure convergence.")
        else: 
            print("No valid runs completed to calculate metrics.")
    else: 
        print("No questions processed.")

    # --- Storing results to JSON and CSV files ---
    json_output_filename = "mmlu_debate_loop_eval_results_final.json"
    csv_output_filename = "mmlu_debate_loop_eval_summary_final.csv"
    try:
        with open(json_output_filename, "w") as f: json.dump(all_results, f, indent=4)
        print(f"\n✅ Successfully saved detailed results to {json_output_filename}")
    except Exception as e: print(f"\n❌ Error saving results to JSON: {e}")
    try:
        with open(csv_output_filename, "w", newline="", encoding="utf-8") as f:
            # ... (CSV writing logic is unchanged)
            fieldnames = [
                "id", "question", "options_A", "options_B", "options_C", "options_D", "options_E", 
                "correct_answer_letter", "full_pipeline_output_details_json"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for res_item in all_results:
                row = {
                    "id": res_item.get("id"),
                    "question": res_item.get("question"),
                    "options_A": res_item.get("options", {}).get("A"),
                    "options_B": res_item.get("options", {}).get("B"),
                    "options_C": res_item.get("options", {}).get("C"),
                    "options_D": res_item.get("options", {}).get("D"),
                    "options_E": res_item.get("options", {}).get("E"),
                    "correct_answer_letter": res_item.get("correct_answer_letter"),
                    "full_pipeline_output_details_json": json.dumps(res_item.get("pipeline_output_details"))
                }
                writer.writerow(row)
        print(f"✅ Successfully saved summary results to {csv_output_filename}")
    except Exception as e: print(f"\n❌ Error saving results to CSV: {e}")