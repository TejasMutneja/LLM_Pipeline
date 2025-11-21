#!/usr/bin/env python3
"""
SQuAD to MCQ Converter

This script converts SQuAD dataset questions into multiple choice questions (MCQs)
by generating plausible distractors using an LLM, then saves them in the format
expected by the existing pipeline.
"""

import os
import json
import random
from typing import List, Dict, Any
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client for Together AI
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("ERROR: TOGETHER_API_KEY not found in environment variables.")
    exit()

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)

MCQ_GENERATION_MODEL = "meta-llama/Llama-3.2-3B-Instruct-Turbo"  # Use serverless model

def generate_mcq_options(context: str, question: str, correct_answer: str, num_distractors: int = 3) -> Dict[str, str]:
    """
    Generate multiple choice options for a SQuAD question using an LLM.
    
    Args:
        context: The passage/context from SQuAD
        question: The question text
        correct_answer: The correct answer from SQuAD
        num_distractors: Number of incorrect options to generate (default: 3)
        
    Returns:
        Dictionary with options A, B, C, D where one is correct and others are distractors
    """
    
    system_prompt = f"""You are an expert test creator. Your task is to create {num_distractors} plausible but incorrect answer options for a reading comprehension question.

Instructions:
1. You will be given a passage, a question, and the correct answer
2. Create {num_distractors} incorrect but plausible alternatives (distractors)
3. The distractors should be:
   - Factually incorrect but reasonable-sounding
   - Similar in format/length to the correct answer
   - Related to the context but not answering the specific question
   - Not obviously wrong at first glance

Format your response as a JSON list of exactly {num_distractors} distractor options.
Example: ["distractor 1", "distractor 2", "distractor 3"]
"""

    user_prompt = f"""Passage: {context}

Question: {question}

Correct Answer: {correct_answer}

Generate {num_distractors} plausible incorrect alternatives:"""

    try:
        response = client.chat.completions.create(
            model=MCQ_GENERATION_MODEL,
            temperature=0.7,
            max_tokens=300,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON response
        try:
            distractors = json.loads(response_text)
            if isinstance(distractors, list) and len(distractors) == num_distractors:
                # Create options dictionary
                options = {}
                letters = ['A', 'B', 'C', 'D']
                
                # Randomly place the correct answer
                correct_position = random.randint(0, num_distractors)
                
                position = 0
                for i in range(num_distractors + 1):
                    if i == correct_position:
                        options[letters[position]] = correct_answer
                    else:
                        distractor_idx = i if i < correct_position else i - 1
                        if distractor_idx < len(distractors):
                            options[letters[position]] = distractors[distractor_idx]
                    position += 1
                
                return options, letters[correct_position]
                
        except json.JSONDecodeError:
            # Fallback: try to extract options from text
            lines = response_text.split('\n')
            distractors = []
            for line in lines:
                line = line.strip().strip('"').strip("'").strip()
                if line and not line.startswith('[') and not line.startswith(']'):
                    distractors.append(line)
            
            if len(distractors) >= num_distractors:
                distractors = distractors[:num_distractors]
                
                options = {}
                letters = ['A', 'B', 'C', 'D']
                correct_position = random.randint(0, num_distractors)
                
                position = 0
                for i in range(num_distractors + 1):
                    if i == correct_position:
                        options[letters[position]] = correct_answer
                    else:
                        distractor_idx = i if i < correct_position else i - 1
                        options[letters[position]] = distractors[distractor_idx]
                    position += 1
                
                return options, letters[correct_position]
    
    except Exception as e:
        print(f"Error generating MCQ options: {e}")
    
    # Fallback: create simple distractors
    options = {
        'A': correct_answer,
        'B': f"Not {correct_answer}",
        'C': f"Alternative to {correct_answer}",
        'D': f"Different from {correct_answer}"
    }
    return options, 'A'

def convert_squad_to_mcq(dataset_name: str = "squad", 
                        split: str = "validation", 
                        num_questions: int = 100,
                        output_file: str = "squad_mcq_questions.json",
                        verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Convert SQuAD dataset questions to MCQ format.
    
    Args:
        dataset_name: Name of the SQuAD dataset to load
        split: Dataset split to use (train/validation)
        num_questions: Number of questions to convert
        output_file: Path to save the converted questions
        verbose: Whether to print progress
        
    Returns:
        List of MCQ questions in the format expected by the pipeline
    """
    
    if verbose:
        print(f"Loading {dataset_name} dataset ({split} split)...")
    
    # Load SQuAD dataset
    try:
        dataset = load_dataset(dataset_name, split=split)
        if len(dataset) < num_questions:
            num_questions = len(dataset)
            print(f"Warning: Dataset only has {len(dataset)} questions. Using all available.")
        
        # Shuffle and select subset
        dataset = dataset.shuffle(seed=42).select(range(num_questions))
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []
    
    mcq_questions = []
    
    for i, item in enumerate(dataset):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing question {i + 1}/{num_questions}...")
        
        context = item['context']
        question = item['question']
        
        # SQuAD answers are in a list format
        if isinstance(item['answers'], dict) and 'text' in item['answers']:
            correct_answer = item['answers']['text'][0] if item['answers']['text'] else "Unknown"
        else:
            correct_answer = str(item['answers']) if item['answers'] else "Unknown"
        
        # Generate MCQ options
        try:
            options, correct_letter = generate_mcq_options(context, question, correct_answer)
            
            # Create MCQ question in the expected format
            mcq_question = {
                "id": f"squad_mcq_{i + 1}",
                "question": f"Context: {context}\n\nQuestion: {question}",
                "options": options,
                "correct_answer_letter": correct_letter,
                "original_squad_data": {
                    "context": context,
                    "question": question,
                    "answer": correct_answer,
                    "id": item.get('id', f'squad_{i}')
                }
            }
            
            mcq_questions.append(mcq_question)
            
        except Exception as e:
            if verbose:
                print(f"Error processing question {i + 1}: {e}")
            continue
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mcq_questions, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"\n✅ Successfully converted {len(mcq_questions)} SQuAD questions to MCQ format")
            print(f"✅ Saved to: {output_file}")
    
    except Exception as e:
        print(f"Error saving to file: {e}")
    
    return mcq_questions

def preview_mcq_questions(mcq_questions: List[Dict[str, Any]], num_preview: int = 3):
    """Preview a few converted MCQ questions."""
    print(f"\n--- Preview of {min(num_preview, len(mcq_questions))} converted MCQ questions ---")
    
    for i, mcq in enumerate(mcq_questions[:num_preview]):
        print(f"\n=== Question {i + 1} ===")
        print(f"ID: {mcq['id']}")
        print(f"Question: {mcq['question'][:200]}...")  # Truncate long questions
        print("Options:")
        for letter, option in mcq['options'].items():
            marker = " ✓" if letter == mcq['correct_answer_letter'] else ""
            print(f"  ({letter}) {option}{marker}")
        print(f"Correct Answer: {mcq['correct_answer_letter']}")

if __name__ == "__main__":
    # Configuration
    DATASET_NAME = "squad"  # or "squad_v2" for SQuAD 2.0
    SPLIT = "validation"    # Use validation split for testing
    NUM_QUESTIONS = 400      # Number of questions to convert
    OUTPUT_FILE = "squad_mcq_questions.json"
    VERBOSE = True
    
    print("=== SQuAD to MCQ Converter ===")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Split: {SPLIT}")
    print(f"Number of questions: {NUM_QUESTIONS}")
    print(f"Output file: {OUTPUT_FILE}")
    print("-" * 50)
    
    # Convert SQuAD questions to MCQ format
    mcq_questions = convert_squad_to_mcq(
        dataset_name=DATASET_NAME,
        split=SPLIT,
        num_questions=NUM_QUESTIONS,
        output_file=OUTPUT_FILE,
        verbose=VERBOSE
    )
    
    if mcq_questions:
        # Preview some questions
        preview_mcq_questions(mcq_questions, num_preview=3)
        
        print(f"\n🎉 Conversion complete! You can now use '{OUTPUT_FILE}' with the modified pipeline.")
        print("\nNext steps:")
        print("1. Run the modified pipeline.py to evaluate the SQuAD MCQ questions")
        print("2. Check the results in the output files")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")