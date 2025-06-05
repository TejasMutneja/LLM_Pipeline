# Block 1: Imports and Setup
import os
import json
import csv
import re
import uuid
from openai import OpenAI # Will be configured for Together AI
from langsmith import traceable
from dotenv import load_dotenv
from datasets import load_dataset

# Load secrets from our .env file
load_dotenv()

# Configure LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# Ensure this project name matches what you have in your .env or set it directly
# If LANGCHAIN_PROJECT is in .env, this line can be commented out.
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "TogetherAI_MMLU_Eval_Shuffled")


# Initialize the OpenAI client to point to Together AI
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("ERROR: TOGETHER_API_KEY not found in environment variables. Please set it in your .env file.")
    exit()

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)
print("Setup complete. Configured to use Together AI for MMLU evaluation with LangSmith.")

# ---

# Block 2: Define the "Worker" Functions with Together AI Model Names
# CRITICAL: These are example model names.
# Check Together AI's documentation for the latest and most suitable model identifiers.
# E.g., Llama 3 Instruct, Mixtral Instruct, Gemma Instruct variants.
LLM1_MODEL = "meta-llama/Llama-3-8b-chat-hf"  # Generator and Corrector
LLM2_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1" # Verifier and Feedback Provider
# Alternative for LLM2 if Mixtral is too large or you prefer Gemma: "google/gemma-2-9b-it"

@traceable
def ask_llm1_to_answer(question_text: str, options_text: str):
    system_prompt = (
        "You are a careful exam-taker. "
        "Given a multiple-choice question plus its options, "
        "return **only one line** formatted exactly as:\n"
        "(<LETTER>) <SHORT JUSTIFICATION>\n"
        "where <LETTER> is A, B, C, D or E. "
        "Do **not** add any extra text before or after."
    )
    full_prompt = f"{question_text}\n{options_text}"
    print(f"  Asking LLM1 ({LLM1_MODEL} via Together AI) to answer:\n{full_prompt[:200]}...") # Truncate for brevity
    response = client.chat.completions.create(
        model=LLM1_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.choices[0].message.content

@traceable
def ask_llm2_to_check(question_text: str, options_text: str, answer_from_llm1: str):
    system_prompt = (
        "You are a verifier. "
        "Respond with **exactly** the single word 'Agree' or 'Disagree' "
        "without punctuation."
    )
    full_prompt = f"{question_text}\n{options_text}"
    print(f"  Asking LLM2 ({LLM2_MODEL} via Together AI) to check answer: {answer_from_llm1[:100]}...") # Truncate
    response = client.chat.completions.create(
        model=LLM2_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question:\n{full_prompt}\n\nProposed answer:\n{answer_from_llm1}"}
        ]
    )
    return response.choices[0].message.content

@traceable
def ask_llm2_for_feedback(question_text: str, options_text: str, incorrect_answer: str):
    system_prompt = (
        "You are a teaching assistant. "
        "The answer above is incorrect. Explain the error **briefly** "
        "and point to the correct reasoning (2–3 sentences)."
    )
    full_prompt = f"{question_text}\n{options_text}"
    print(f"  Asking LLM2 ({LLM2_MODEL} via Together AI) for feedback on: {incorrect_answer[:100]}...") # Truncate
    response = client.chat.completions.create(
        model=LLM2_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question:\n{full_prompt}\n\nIncorrect answer:\n{incorrect_answer}"}
        ]
    )
    return response.choices[0].message.content

@traceable
def ask_llm1_to_retry_with_feedback(question_text: str, options_text: str, feedback: str):
    system_prompt = (
        "You previously answered a multiple-choice question incorrectly. "
        "Use the feedback to give a **new single-line answer** exactly in the "
        "format '(X) justification', with X the correct option letter."
    )
    full_prompt = f"{question_text}\n{options_text}"
    print(f"  Asking LLM1 ({LLM1_MODEL} via Together AI) to retry with feedback: {feedback[:100]}...") # Truncate
    response = client.chat.completions.create(
        model=LLM1_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original question:\n{full_prompt}\n\nFeedback:\n{feedback}"}
        ]
    )
    return response.choices[0].message.content

# ---

# Block 3: Orchestrator function (logic unchanged)
@traceable(run_type="chain")
def mmlu_pipeline(question_data: dict):
    question_text = question_data["question"]
    options_list  = [f"({k}) {v}" for k, v in question_data["options"].items()]
    options_text  = "\n".join(options_list)

    llm1_answer   = ask_llm1_to_answer(question_text, options_text)
    # Adding a print statement here for debugging the raw output of LLM1
    # print(f"DEBUG: Raw LLM1 Answer: '{llm1_answer}'") 
    verification  = ask_llm2_to_check(question_text, options_text, llm1_answer)
    # Adding a print statement here for debugging the raw output of LLM2 Check
    # print(f"DEBUG: Raw LLM2 Verification: '{verification}'")


    pipeline_result = {
        "llm1_initial_answer": llm1_answer,
        "llm2_verification"  : verification,
        "llm2_feedback"      : None,
        "llm1_corrected_answer": None,
        "final_answer_for_evaluation": llm1_answer # Default to initial answer
    }

    # Ensure verification output is just "Agree" or "Disagree"
    processed_verification = verification.strip().capitalize()

    if processed_verification == "Agree":
        print(f"\nLLM1's Initial Answer:\n---\n{llm1_answer}\n---")
        print(f"LLM2's Verification: {processed_verification}")
        print("\nPipeline finished. Models in agreement.")
        # final_answer_for_evaluation is already llm1_answer
    elif processed_verification == "Disagree":
        print(f"\nLLM1's Initial Answer:\n---\n{llm1_answer}\n---")
        print(f"LLM2's Verification: {processed_verification}")
        print("\nModels disagree. Starting feedback loop...")
        feedback = ask_llm2_for_feedback(question_text, options_text, llm1_answer)
        print(f"\nFeedback from LLM2:\n---\n{feedback}\n---")
        pipeline_result["llm2_feedback"] = feedback

        final_answer = ask_llm1_to_retry_with_feedback(question_text, options_text, feedback)
        # print(f"DEBUG: Raw LLM1 Corrected Answer: '{final_answer}'") # Debugging corrected answer
        print("\nPipeline finished after correction.")
        pipeline_result["llm1_corrected_answer"]      = final_answer
        pipeline_result["final_answer_for_evaluation"] = final_answer
    else: # Handle unexpected verifier output
        print(f"\nLLM1's Initial Answer:\n---\n{llm1_answer}\n---")
        print(f"LLM2's Verification (Unexpected): '{verification}'")
        print("Warning: LLM2 verifier returned an unexpected response. Treating as disagreement for safety.")
        print("Proceeding with feedback loop...")
        # Fallback: Treat unexpected output as a disagreement to get feedback
        feedback = ask_llm2_for_feedback(question_text, options_text, llm1_answer)
        print(f"\nFeedback from LLM2:\n---\n{feedback}\n---")
        pipeline_result["llm2_feedback"] = feedback
        pipeline_result["llm2_verification"] = f"Disagree (Interpreted from: {verification})" # Log original

        final_answer = ask_llm1_to_retry_with_feedback(question_text, options_text, feedback)
        print("\nPipeline finished after correction (due to unexpected verification).")
        pipeline_result["llm1_corrected_answer"]      = final_answer
        pipeline_result["final_answer_for_evaluation"] = final_answer

    return pipeline_result
# ---

# Block 4: Letter-extraction helper (unchanged)
def get_letter_from_output(text_output: str | None):
    if not text_output:
        return None
    m = re.match(r'^\s*\(([A-Ea-e])\)', text_output.strip())
    if m:
        return m.group(1).upper()
    m2 = re.match(r'^\s*([A-Ea-e])\.?\s*$', text_output.strip())
    if m2:
        return m2.group(1).upper()
    # print(f"Warning: Could not reliably parse letter from output: '{text_output[:100]}...'") # Keep this if you see parsing issues
    return None

# ---

# Block 5: Dataset Loading, Evaluation Loop, Metrics, and Saving Results
if __name__ == "__main__":
    print("--- Loading MMLU dataset from Hugging Face ---")
    
    hf_dataset = None
    MMLU_SUBJECT = "high_school_us_history" 
    NUM_QUESTIONS_TO_PROCESS = 100 # Keep small for initial testing with API calls
    SHUFFLE_SEED = 42 # Fixed seed for reproducible shuffling

    try:
        # Load the full subject dataset first
        full_subject_dataset = load_dataset("cais/mmlu", MMLU_SUBJECT, split="test")
        # Shuffle the dataset with a fixed seed
        shuffled_dataset = full_subject_dataset.shuffle(seed=SHUFFLE_SEED)
        
        if len(shuffled_dataset) < NUM_QUESTIONS_TO_PROCESS:
            print(f"Warning: Requested {NUM_QUESTIONS_TO_PROCESS} questions, but shuffled subject '{MMLU_SUBJECT}' only has {len(shuffled_dataset)}. Processing all available.")
            hf_dataset = shuffled_dataset.select(range(len(shuffled_dataset)))
        else:
            # Select a consistent slice from the shuffled dataset
            hf_dataset = shuffled_dataset.select(range(NUM_QUESTIONS_TO_PROCESS))
        print(f"Loaded and shuffled {len(hf_dataset)} questions from Hugging Face MMLU ('{MMLU_SUBJECT}' subset, seed={SHUFFLE_SEED}).")

    except Exception as e:
        print(f"Failed to load dataset from Hugging Face: {e}")
        print("Please ensure 'datasets' library is installed (`pip install datasets`) and you have internet access.")
        print("Exiting.")
        exit()

    print(f"--- Starting MMLU Pipeline Evaluation for {len(hf_dataset)} questions using Together AI ---")
    
    all_results = []
    option_letters = ["A", "B", "C", "D", "E"] 

    for i, item in enumerate(hf_dataset):
        question_text = item["question"]
        current_options = {}
        if len(item["choices"]) >= 1:
            for choice_idx, choice_text in enumerate(item["choices"]):
                if choice_idx < len(option_letters):
                    current_options[option_letters[choice_idx]] = choice_text
                else:
                    break 
        else:
            print(f"Skipping item {i+1} due to no choices: {item}")
            continue
        if not current_options:
            print(f"Skipping item {i+1} as no valid options could be formed: {item}")
            continue

        correct_answer_index = item["answer"]
        correct_answer_letter = option_letters[correct_answer_index] if 0 <= correct_answer_index < len(current_options) else "N/A"

        question_data_for_pipeline = {
            "id": f"hf_mmlu_{MMLU_SUBJECT}_{SHUFFLE_SEED}_{i+1}", # Include seed in ID for uniqueness
            "question": question_text,
            "options": current_options,
            "correct_answer_letter": correct_answer_letter
        }
        
        print(f"\n--- Processing Question {i+1}/{len(hf_dataset)} (ID: {question_data_for_pipeline['id']}) ---")
        
        try:
            pipeline_output = mmlu_pipeline(question_data_for_pipeline)
        except Exception as e:
            print(f"!!!!!!!! ERROR processing pipeline for question ID {question_data_for_pipeline['id']}: {e} !!!!!!!!")
            pipeline_output = { # Log a failure structure
                "llm1_initial_answer": f"ERROR: {e}",
                "llm2_verification": "ERROR",
                "llm2_feedback": None,
                "llm1_corrected_answer": None,
                "final_answer_for_evaluation": "ERROR"
            }
        
        print("\n===================================")
        print(f"✅ Final Answer for Question ID {question_data_for_pipeline['id']}:\n{pipeline_output['final_answer_for_evaluation']}")
        print(f"   Correct Answer Letter: {correct_answer_letter}")
        print("===================================")
        
        result_summary = {**question_data_for_pipeline, "pipeline_output_details": pipeline_output}
        all_results.append(result_summary)

    print("\n--- Evaluation Loop Finished ---")
    print(f"Processed {len(all_results)} questions.")

    # --- Calculate Evaluation Metrics ---
    total_questions_attempted = len(all_results) # Use all_results as some might have errored
    llm1_initial_correct_count = 0
    agreement_count = 0
    final_pipeline_correct_count = 0
    disagreements_count = 0
    corrected_after_disagreement_count = 0
    error_in_pipeline_count = 0

    if total_questions_attempted > 0:
        for res in all_results:
            # Check if this question resulted in an error during pipeline processing
            if res['pipeline_output_details']['final_answer_for_evaluation'] == "ERROR":
                error_in_pipeline_count +=1
                continue # Skip metrics for errored runs

            llm1_initial_answer_letter = get_letter_from_output(res['pipeline_output_details']['llm1_initial_answer'])
            final_answer_letter = get_letter_from_output(res['pipeline_output_details']['final_answer_for_evaluation'])
            correct_letter = res['correct_answer_letter']

            if llm1_initial_answer_letter == correct_letter:
                llm1_initial_correct_count += 1
            
            verification_text = res['pipeline_output_details']['llm2_verification']
            if "Agree" in verification_text.strip().capitalize(): # Check for Agree even if it was an interpreted one
                agreement_count += 1
            elif verification_text != "ERROR": # Only count as disagreement if not an error
                disagreements_count += 1
                # Check for correction only if there was a disagreement and no error in correction
                if res['pipeline_output_details']['llm1_corrected_answer'] is not None and \
                   res['pipeline_output_details']['llm1_corrected_answer'] != "ERROR":
                    corrected_answer_letter = get_letter_from_output(res['pipeline_output_details']['llm1_corrected_answer'])
                    if corrected_answer_letter == correct_letter and llm1_initial_answer_letter != correct_letter:
                        corrected_after_disagreement_count += 1
            
            if final_answer_letter == correct_letter:
                final_pipeline_correct_count += 1
        
        # Adjust total questions for metrics if some runs errored out
        valid_runs_for_metrics = total_questions_attempted - error_in_pipeline_count

        print("\n--- Evaluation Metrics ---")
        if error_in_pipeline_count > 0:
            print(f"Pipeline Errors: {error_in_pipeline_count}/{total_questions_attempted}")
        
        if valid_runs_for_metrics > 0:
            print(f"LLM1 Initial Accuracy: {llm1_initial_correct_count / valid_runs_for_metrics:.2%} ({llm1_initial_correct_count}/{valid_runs_for_metrics})")
            print(f"Agreement Rate (LLM2 agreed with LLM1): {agreement_count / valid_runs_for_metrics:.2%} ({agreement_count}/{valid_runs_for_metrics})")
            print(f"Final Pipeline Accuracy: {final_pipeline_correct_count / valid_runs_for_metrics:.2%} ({final_pipeline_correct_count}/{valid_runs_for_metrics})")
            if disagreements_count > 0:
                print(f"Correction Success Rate (when LLM2 disagreed): {corrected_after_disagreement_count / disagreements_count:.2%} ({corrected_after_disagreement_count}/{disagreements_count})")
            else:
                print("No valid disagreements occurred to measure correction rate.")
        else:
            print("No valid runs completed to calculate metrics.")
            
    else:
        print("No questions processed to calculate metrics.")

    # --- Storing results to JSON file ---
    json_output_filename = "mmlu_togetherai_eval_results.json"
    try:
        with open(json_output_filename, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\n✅ Successfully saved detailed results to {json_output_filename}")
    except Exception as e:
        print(f"\n❌ Error saving results to JSON: {e}")

    # --- Storing key results to CSV file ---
    csv_output_filename = "mmlu_togetherai_eval_summary.csv"
    try:
        fieldnames = [
            "id", "question", 
            "options_A", "options_B", "options_C", "options_D", "options_E",
            "correct_answer_letter",
            "llm1_initial_answer_text", "llm1_initial_parsed_letter",
            "llm2_verification_text",
            "llm2_feedback_text",
            "llm1_corrected_answer_text", "llm1_corrected_parsed_letter",
            "final_answer_for_evaluation_text", "final_parsed_letter",
            "full_pipeline_output_details_json"
        ]
        with open(csv_output_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for res_item in all_results:
                # Prepare row data, handling potential errors in pipeline_output_details
                pipeline_details = res_item.get("pipeline_output_details", {})
                row = {
                    "id": res_item.get("id"),
                    "question": res_item.get("question"),
                    "options_A": res_item.get("options", {}).get("A"),
                    "options_B": res_item.get("options", {}).get("B"),
                    "options_C": res_item.get("options", {}).get("C"),
                    "options_D": res_item.get("options", {}).get("D"),
                    "options_E": res_item.get("options", {}).get("E"),
                    "correct_answer_letter": res_item.get("correct_answer_letter"),
                    "llm1_initial_answer_text": pipeline_details.get("llm1_initial_answer"),
                    "llm1_initial_parsed_letter": get_letter_from_output(pipeline_details.get("llm1_initial_answer")),
                    "llm2_verification_text": pipeline_details.get("llm2_verification"),
                    "llm2_feedback_text": pipeline_details.get("llm2_feedback"),
                    "llm1_corrected_answer_text": pipeline_details.get("llm1_corrected_answer"),
                    "llm1_corrected_parsed_letter": get_letter_from_output(pipeline_details.get("llm1_corrected_answer")),
                    "final_answer_for_evaluation_text": pipeline_details.get("final_answer_for_evaluation"),
                    "final_parsed_letter": get_letter_from_output(pipeline_details.get("final_answer_for_evaluation")),
                    "full_pipeline_output_details_json": json.dumps(pipeline_details)
                }
                writer.writerow(row)
        print(f"✅ Successfully saved summary results to {csv_output_filename}")
    except Exception as e:
        print(f"\n❌ Error saving results to CSV: {e}")