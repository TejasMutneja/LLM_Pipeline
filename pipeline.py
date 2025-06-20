# Block 1: Imports and Setup
import os
import json
import csv
import re
import uuid
from openai import OpenAI
from langsmith import traceable
from dotenv import load_dotenv
from datasets import load_dataset

# Load secrets from our .env file
load_dotenv()

# Configure LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "TogetherAI_Parser_Fix_Eval")

# Initialize the OpenAI client to point to Together AI
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("ERROR: TOGETHER_API_KEY not found in environment variables. Please set it in your .env file.")
    exit()

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)
print("Setup complete. Configured with enhanced parsing for Together AI evaluation.")

# ---

# Block 2: Worker Functions with Enhanced Prompts
LLM1_MODEL = "meta-llama/Llama-3-8b-chat-hf"
LLM2_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

@traceable
def ask_llm1_to_answer(question_text: str, options_text: str, verbose: bool = False):
    ### --- UPDATED PROMPT with Few-Shot Example --- ###
    system_prompt = (
        "You are a careful exam-taker. "
        "Given a multiple-choice question plus its options, "
        "first provide a short justification for your answer, "
        "and then on the SAME LINE, provide the letter choice in parentheses.\n"
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
    # This prompt is intentionally strict and unchanged.
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
    # This prompt is unchanged.
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
    ### --- UPDATED PROMPT with Few-Shot Example --- ###
    system_prompt = (
        "You previously answered incorrectly. Use the provided feedback to re-evaluate the question. "
        "First, write a new, short justification based on the feedback, "
        "then on the SAME LINE, provide the new option letter in parentheses.\n"
        "Your entire response must be a SINGLE LINE formatted exactly as:\n"
        "<NEW JUSTIFICATION> (<LETTER>)\n\n"
        "For example:\n"
        "After re-evaluating, the key author was indeed Thomas Jefferson. (A)"
    )
    full_prompt = f"{question_text}\n{options_text}"
    if verbose:
        print(f"  Asking LLM1 ({LLM1_MODEL}) to retry with CoT prompt...")
    response = client.chat.completions.create(
        model=LLM1_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Original question:\n{full_prompt}\n\nFeedback on your mistake:\n{feedback}"}]
    )
    return response.choices[0].message.content

# ---

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
        
        if processed_verification != "Disagree":
            pipeline_result["llm2_verification"] = f"Disagree (Interpreted from: {verification})"

        final_answer = ask_llm1_to_retry_with_feedback(question_text, options_text, feedback, verbose=verbose)
        if verbose: print("\nPipeline finished after correction.")
        pipeline_result["llm1_corrected_answer"] = final_answer
        pipeline_result["final_answer_for_evaluation"] = final_answer

    return pipeline_result

# ---

# Block 4: Letter-extraction helper (UPDATED to be more robust)
def get_letter_from_output(text_output: str | None):
    if not text_output: return None
    
    clean_text = text_output.strip()

    # Priority 1: Look for (<LETTER>) at the end of the string.
    m1 = re.search(r'\(([A-Ea-e])\)\s*$', clean_text)
    if m1: return m1.group(1).upper()

    # Priority 2: Look for (<LETTER>) anywhere in the string.
    m2 = re.search(r'\(([A-Ea-e])\)', clean_text)
    if m2: return m2.group(1).upper()

    # Priority 3: Look for a single letter as the entire output.
    m3 = re.match(r'^\s*([A-Ea-e])\.?\s*$', clean_text)
    if m3: return m3.group(1).upper()

    # Priority 4: Look for phrases like "is A", "is B.", "is C," at the end of the string.
    m4 = re.search(r'is\s+([A-Ea-e])[\.\s,]*$', clean_text, re.IGNORECASE)
    if m4: return m4.group(1).upper()
    
    # This warning is now a last resort.
    print(f"Warning: Could not reliably parse letter from output: '{text_output[:100]}...'")
    return None

# ---

# Block 5: Dataset Loading, Evaluation Loop, Metrics, and Saving Results (Unchanged)
if __name__ == "__main__":
    # --- SCRIPT CONFIGURATION ---
    VERBOSE = False 
    MMLU_SUBJECT = "high_school_us_history"
    NUM_QUESTIONS_TO_PROCESS = 100
    SHUFFLE_SEED = 42
    # ---------------------------

    print("--- Loading MMLU dataset from Hugging Face ---")
    hf_dataset = None
    try:
        full_subject_dataset = load_dataset("cais/mmlu", MMLU_SUBJECT, split="test")
        shuffled_dataset = full_subject_dataset.shuffle(seed=SHUFFLE_SEED)
        if len(shuffled_dataset) < NUM_QUESTIONS_TO_PROCESS: hf_dataset = shuffled_dataset
        else: hf_dataset = shuffled_dataset.select(range(NUM_QUESTIONS_TO_PROCESS))
        print(f"Loaded and shuffled {len(hf_dataset)} questions from Hugging Face MMLU ('{MMLU_SUBJECT}' subset, seed={SHUFFLE_SEED}).")
    except Exception as e:
        print(f"Failed to load dataset from Hugging Face: {e}\nExiting.")
        exit()

    print(f"--- Starting MMLU Pipeline Evaluation for {len(hf_dataset)} questions using Together AI ---")
    
    all_results, option_letters = [], ["A", "B", "C", "D", "E"]

    for i, item in enumerate(hf_dataset):
        if VERBOSE: print(f"\n--- Processing Question {i+1}/{len(hf_dataset)} (ID: hf_mmlu_{MMLU_SUBJECT}_{SHUFFLE_SEED}_{i+1}) ---")
        else: print(f"Processing question {i+1}/{len(hf_dataset)}...", end='\r')

        current_options = {option_letters[idx]: text for idx, text in enumerate(item["choices"]) if idx < len(option_letters)}
        if not current_options:
            if VERBOSE: print(f"Skipping item {i+1} due to no choices.")
            continue
        
        correct_answer_index = item["answer"]
        correct_answer_letter = option_letters[correct_answer_index] if 0 <= correct_answer_index < len(current_options) else "N/A"

        question_data_for_pipeline = {"id": f"hf_mmlu_{MMLU_SUBJECT}_{SHUFFLE_SEED}_{i+1}", "question": item["question"], "options": current_options, "correct_answer_letter": correct_answer_letter}
        
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

    # --- Calculate Evaluation Metrics ---
    # ... (This logic is unchanged, but should now receive better data) ...
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
        print("\n--- Evaluation Metrics ---")
        if error_in_pipeline_count > 0: print(f"Pipeline Errors: {error_in_pipeline_count}/{total_questions_attempted}")
        if valid_runs_for_metrics > 0:
            print(f"LLM1 Initial Accuracy: {llm1_initial_correct_count / valid_runs_for_metrics:.2%} ({llm1_initial_correct_count}/{valid_runs_for_metrics})")
            print(f"Agreement Rate (LLM2 agreed with LLM1): {agreement_count / valid_runs_for_metrics:.2%} ({agreement_count}/{valid_runs_for_metrics})")
            print(f"Final Pipeline Accuracy: {final_pipeline_correct_count / valid_runs_for_metrics:.2%} ({final_pipeline_correct_count}/{valid_runs_for_metrics})")
            if disagreements_count > 0: print(f"Correction Success Rate (when LLM2 disagreed): {corrected_after_disagreement_count / disagreements_count:.2%} ({corrected_after_disagreement_count}/{disagreements_count})")
            else: print("No valid disagreements occurred to measure correction rate.")
        else: print("No valid runs completed to calculate metrics.")
    else: print("No questions processed to calculate metrics.")

    # --- Storing results to JSON and CSV files ---
    json_output_filename = "mmlu_togetherai_parser_fix_eval.json"
    csv_output_filename = "mmlu_togetherai_parser_fix_summary.csv"
    try:
        with open(json_output_filename, "w") as f: json.dump(all_results, f, indent=4)
        print(f"\n✅ Successfully saved detailed results to {json_output_filename}")
    except Exception as e:
        print(f"\n❌ Error saving results to JSON: {e}")
    try:
        fieldnames = ["id", "question", "options_A", "options_B", "options_C", "options_D", "options_E", "correct_answer_letter", "llm1_initial_answer_text", "llm1_initial_parsed_letter", "llm2_verification_text", "llm2_feedback_text", "llm1_corrected_answer_text", "llm1_corrected_parsed_letter", "final_answer_for_evaluation_text", "final_parsed_letter", "full_pipeline_output_details_json"]
        with open(csv_output_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for res_item in all_results:
                pipeline_details = res_item.get("pipeline_output_details", {})
                row = {"id": res_item.get("id"), "question": res_item.get("question"), "options_A": res_item.get("options", {}).get("A"), "options_B": res_item.get("options", {}).get("B"), "options_C": res_item.get("options", {}).get("C"), "options_D": res_item.get("options", {}).get("D"), "options_E": res_item.get("options", {}).get("E"), "correct_answer_letter": res_item.get("correct_answer_letter"), "llm1_initial_answer_text": pipeline_details.get("llm1_initial_answer"), "llm1_initial_parsed_letter": get_letter_from_output(pipeline_details.get("llm1_initial_answer")), "llm2_verification_text": pipeline_details.get("llm2_verification"), "llm2_feedback_text": pipeline_details.get("llm2_feedback"), "llm1_corrected_answer_text": pipeline_details.get("llm1_corrected_answer"), "llm1_corrected_parsed_letter": get_letter_from_output(pipeline_details.get("llm1_corrected_answer")), "final_answer_for_evaluation_text": pipeline_details.get("final_answer_for_evaluation"), "final_parsed_letter": get_letter_from_output(pipeline_details.get("final_answer_for_evaluation")), "full_pipeline_output_details_json": json.dumps(pipeline_details)}
                writer.writerow(row)
        print(f"✅ Successfully saved summary results to {csv_output_filename}")
    except Exception as e:
        print(f"\n❌ Error saving results to CSV: {e}")