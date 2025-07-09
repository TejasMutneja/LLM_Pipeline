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
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "TogetherAI_Iterative_Debate_Eval")

# Initialize the OpenAI client to point to Together AI
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("ERROR: TOGETHER_API_KEY not found in environment variables.")
    exit()

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)
print("Setup complete. Configured with iterative discussion loop.")

# ---

# Block 2: Worker Functions (Updated for Loop)
LLM1_MODEL = "meta-llama/Llama-3-8b-chat-hf"
LLM2_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

@traceable
def ask_llm1_to_answer(question_text: str, options_text: str, verbose: bool = False):
    # This initial prompt is unchanged
    system_prompt = (
        "You are an expert exam-taker. Your task is to select the best possible answer from the given options. "
        "First, provide a short justification for your choice, and then on the SAME LINE, provide the letter choice in parentheses.\n"
        "Your entire response must be a SINGLE LINE formatted exactly as:\n"
        "<SHORT JUSTIFICATION> (<LETTER>)"
    )
    full_prompt = f"{question_text}\n{options_text}"
    if verbose: print(f"  Asking LLM1 ({LLM1_MODEL}) for initial answer...")
    response = client.chat.completions.create(model=LLM1_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_prompt}])
    return response.choices[0].message.content

@traceable
def ask_llm2_to_verify_and_justify(question_text: str, options_text: str, answer_from_llm1: str, verbose: bool = False):
    # This merged function is unchanged
    system_prompt = (
        "You are a meticulous and highly critical verifier. Your task is to evaluate a proposed answer to a multiple-choice question. "
        "You MUST respond with a valid JSON object containing two keys: \"verdict\" and \"justification\".\n\n"
        "- The \"verdict\" key must be EXACTLY the string \"Agree\" or \"Disagree\".\n"
        "- 'Agree' only if the proposed answer's option and justification are both unequivocally correct.\n"
        "- 'Disagree' in all other cases. If you disagree, your justification should explain the error.\n"
        "- The \"justification\" key must contain a brief, 1-2 sentence explanation for your verdict."
    )
    full_prompt = f"{question_text}\n{options_text}"
    if verbose: print(f"  Asking LLM2 ({LLM2_MODEL}) to verify and justify...")
    response = client.chat.completions.create(
        model=LLM2_MODEL, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Question:\n{full_prompt}\n\nProposed answer:\n{answer_from_llm1}"}]
    )
    try:
        json_response = json.loads(response.choices[0].message.content)
        if "verdict" in json_response and "justification" in json_response: return json_response
        else: return {"verdict": "Disagree", "justification": "LLM2 failed to produce valid JSON format."}
    except (json.JSONDecodeError, TypeError):
        return {"verdict": "Disagree", "justification": "LLM2 returned non-JSON output."}

### --- PROMPT MODIFIED TO INCLUDE PREVIOUS ANSWER --- ###
@traceable
def ask_llm1_to_retry_with_feedback(question_text: str, options_text: str, feedback: str, previous_answer: str, verbose: bool = False):
    system_prompt = (
        "You are an assistant correcting your own mistake on a multiple-choice question. "
        "You will be given your previous incorrect answer and feedback from an expert verifier. "
        "Use the feedback to re-evaluate the question and provide a new, corrected answer. "
        "First, write a new justification, then on the SAME LINE, provide the new option letter in parentheses.\n"
        "Your entire response must be a SINGLE LINE formatted exactly as:\n"
        "<NEW JUSTIFICATION> (<LETTER>)"
    )
    full_prompt = f"{question_text}\n{options_text}"
    user_prompt = (
        f"Original question:\n{full_prompt}\n\n"
        f"Your previous incorrect answer was:\n{previous_answer}\n\n"
        f"Feedback on your mistake:\n{feedback}\n\n"
        "Please provide the new, corrected answer."
    )
    if verbose: print(f"  Asking LLM1 ({LLM1_MODEL}) to retry...")
    response = client.chat.completions.create(model=LLM1_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    return response.choices[0].message.content

# ---

# Block 3: Orchestrator function (UPDATED WITH ITERATIVE LOOP)
@traceable(run_type="chain")
def mmlu_pipeline(question_data: dict, verbose: bool = False):
    MAX_ROUNDS = 3  # Set the maximum number of back-and-forth turns
    question_text = question_data["question"]
    options_list  = [f"({k}) {v}" for k, v in question_data["options"].items()]
    options_text  = "\n".join(options_list)

    # Initial Answer
    current_answer = ask_llm1_to_answer(question_text, options_text, verbose=verbose)
    initial_answer = current_answer
    
    discussion_history = []

    for round_num in range(MAX_ROUNDS):
        if verbose: print(f"\n--- Round {round_num + 1} ---")
        if verbose: print(f"LLM1's current answer:\n---\n{current_answer}\n---")

        # LLM2 Verifies the current answer
        verification_output = ask_llm2_to_verify_and_justify(question_text, options_text, current_answer, verbose=verbose)
        verdict = verification_output.get("verdict", "Disagree").strip().capitalize()
        justification = verification_output.get("justification", "No justification provided.")

        discussion_history.append({
            "round": round_num + 1,
            "llm1_answer": current_answer,
            "llm2_verdict": verdict,
            "llm2_justification": justification
        })

        if verbose:
            print(f"LLM2's Verification: {verdict}")
            print(f"LLM2's Justification: {justification}")

        # If they agree, the loop is over
        if verdict == "Agree":
            if verbose: print("\nPipeline finished. Models in agreement.")
            break

        # If they disagree and we haven't reached the max number of rounds, retry
        if round_num < MAX_ROUNDS - 1:
            if verbose: print("\nModels disagree. Continuing discussion...")
            feedback = justification # The justification for disagreeing is the feedback
            previous_answer = current_answer # The answer that was just disagreed with
            current_answer = ask_llm1_to_retry_with_feedback(question_text, options_text, feedback, previous_answer, verbose=verbose)
        else:
            if verbose: print("\nMax rounds reached. Pipeline finished.")
    
    # Final structured result
    pipeline_result = {
        "llm1_initial_answer": initial_answer,
        "final_answer_for_evaluation": current_answer,
        "discussion_history": discussion_history
    }
    
    # For backward compatibility with metrics, add these top-level keys
    last_round = discussion_history[-1]
    pipeline_result["llm2_verification"] = last_round["llm2_verdict"]
    if last_round["llm2_verdict"] == "Disagree":
        pipeline_result["llm2_feedback"] = last_round["llm2_justification"]
    # Check if a correction was made
    if len(discussion_history) > 1:
        pipeline_result["llm1_corrected_answer"] = current_answer
    
    return pipeline_result

# ---

# Block 4: Letter-extraction helper (Unchanged)
def get_letter_from_output(text_output: str | None):
    # ... (code is unchanged, but you can copy it from my previous response if needed)
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

# Block 5: Dataset Loading, Evaluation Loop, Metrics, and Saving Results (Unchanged)
if __name__ == "__main__":
    # --- SCRIPT CONFIGURATION ---
    VERBOSE = False 
    DATASET_NAME = "cais/mmlu"
    DATASET_CONFIG = "all"
    NUM_QUESTIONS_TO_PROCESS = 1000
    SHUFFLE_SEED = 42
    # ---------------------------

    # ... (The rest of Block 5 is exactly the same as before) ...
    # It will work correctly with the updated pipeline logic.
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

    print(f"--- Starting Pipeline Evaluation using Together AI ---")
    
    all_results, option_letters = [], ["A", "B", "C", "D", "E"]

    for i, item in enumerate(hf_dataset):
        if not VERBOSE:
            print(f"Processing question {i+1}/{len(hf_dataset)}...", end='\r')
        
        question_text = item["question"]
        current_options = {option_letters[idx]: text for idx, text in enumerate(item["choices"]) if idx < len(option_letters)}
        if not current_options:
            if VERBOSE: print(f"Skipping item {i+1} due to no choices.")
            continue
        
        correct_answer_index = item["answer"]
        correct_answer_letter = option_letters[correct_answer_index] if 0 <= correct_answer_index < len(current_options) else "N/A"

        question_data_for_pipeline = {"id": f"hf_{DATASET_CONFIG}_{SHUFFLE_SEED}_{i+1}", "question": question_text, "options": current_options, "correct_answer_letter": correct_answer_letter}
        
        if VERBOSE: print(f"\n--- Processing Question {i+1}/{len(hf_dataset)} (ID: {question_data_for_pipeline['id']}) ---")
        
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

    print("\n" + "="*50)
    print("--- Evaluation Loop Finished ---")
    print(f"Processed {len(all_results)} questions.")

    # --- Calculate Evaluation Metrics and Sigma ---
    # The existing metrics logic will still work with the updated pipeline output,
    # as we've preserved the key fields it relies on.
    total_questions_attempted = len(all_results)
    llm1_initial_correct_count, agreement_count, final_pipeline_correct_count = 0, 0, 0
    disagreements_count, corrected_after_disagreement_count, error_in_pipeline_count = 0, 0, 0

    if total_questions_attempted > 0:
        for res in all_results:
            if res['pipeline_output_details'].get('final_answer_for_evaluation') == "ERROR":
                error_in_pipeline_count +=1; continue
            
            # Metrics based on initial and final states
            llm1_initial_answer_letter = get_letter_from_output(res['pipeline_output_details']['llm1_initial_answer'])
            final_answer_letter = get_letter_from_output(res['pipeline_output_details']['final_answer_for_evaluation'])
            correct_letter = res['correct_answer_letter']
            
            if llm1_initial_answer_letter == correct_letter: llm1_initial_correct_count += 1
            if final_answer_letter == correct_letter: final_pipeline_correct_count += 1

            # Metrics based on the discussion process
            history = res['pipeline_output_details'].get('discussion_history', [])
            if history:
                first_round_verdict = history[0].get('llm2_verdict')
                if first_round_verdict == "Agree":
                    agreement_count += 1
                elif first_round_verdict == "Disagree":
                    disagreements_count += 1
                    # Check if a correction happened and was successful
                    if llm1_initial_answer_letter != correct_letter and final_answer_letter == correct_letter:
                         corrected_after_disagreement_count += 1
        
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
            
            # The agreement/disagreement metrics now refer to the first round of discussion
            print(f"Initial Agreement Rate: {agreement_count / valid_runs_for_metrics:.2%} ({agreement_count}/{valid_runs_for_metrics})")
            if disagreements_count > 0:
                print(f"Correction Success Rate (on first disagreement): {corrected_after_disagreement_count / disagreements_count:.2%} ({corrected_after_disagreement_count}/{disagreements_count})")
            else:
                print("No initial disagreements occurred to measure correction rate.")

        else: print("No valid runs completed to calculate metrics.")
    else: print("No questions processed.")

    # --- Storing results to JSON and CSV files ---
    json_output_filename = "mmlu_iterative_debate_eval_results.json"
    csv_output_filename = "mmlu_iterative_debate_eval_summary.csv"
    # ... (The file saving logic remains the same) ...