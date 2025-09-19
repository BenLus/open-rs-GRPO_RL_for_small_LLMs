#!/usr/bin/env python3
"""
Script to crawl evaluation logs and create a summary of results for each model.

This script processes evaluation results from the logs/evals folder for three model types:
1. Exp3Ben (with iterations)
2. Qwen3_trained (with iterations) 
3. Qwen3_orig (single results, no iterations)

It creates a dictionary structure that maps:
- Model type -> Iteration number -> Task name -> Results
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict


def find_json_files(directory: Path) -> List[Path]:
    """Recursively find all JSON files in a directory."""
    json_files = []
    if directory.exists() and directory.is_dir():
        for item in directory.rglob("*.json"):
            json_files.append(item)
    return json_files


def extract_task_results(json_file: Path) -> Dict[str, Any]:
    """
    Extract task name and results from a JSON file.
    
    Returns:
        Dictionary with task name as key and results['all'] as value
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {}
        
        # Get config_tasks to find task names
        config_tasks = data.get('config_tasks', {})
        
        for task_key, task_config in config_tasks.items():
            task_name = task_config.get('name')
            if task_name:
                # Get results for this task
                all_results = data.get('results', {}).get('all', {})
                if all_results:
                    results[task_name] = all_results
        
        return results
        
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return {}


def extract_iteration_number(folder_name: str) -> int:
    """Extract iteration number from folder name like 'Exp3_100_Exp3Ben'."""
    # Look for pattern like digits after first underscore
    match = re.search(r'_(\d+)_', folder_name)
    if match:
        return int(match.group(1))
    return 0


def process_model_folders(evals_dir: Path, model_suffix: str) -> Dict[int, Dict[str, Any]]:
    """
    Process all folders for a specific model type and return iteration -> results mapping.
    
    Args:
        evals_dir: Path to the evals directory
        model_suffix: The suffix to match (e.g., 'Exp3Ben', 'Qwen3_trained')
    
    Returns:
        Dictionary mapping iteration numbers to results dictionaries
    """
    model_results = {}
    
    # Find all folders ending with the model suffix
    for folder in evals_dir.iterdir():
        if folder.is_dir() and folder.name.endswith(model_suffix):
            iteration_num = extract_iteration_number(folder.name)
            
            # Find all JSON files in this folder
            json_files = find_json_files(folder)
            
            # Aggregate results from all JSON files for this iteration
            iteration_results = {}
            
            for json_file in json_files:
                task_results = extract_task_results(json_file)
                iteration_results.update(task_results)
            
            if iteration_results:
                model_results[iteration_num] = iteration_results
                print(f"Processed {folder.name}: {len(iteration_results)} tasks")
    
    return model_results


def process_qwen3_orig(evals_dir: Path) -> Dict[str, Any]:
    """
    Process Qwen3_orig folder which has no iterations.
    
    Returns:
        Dictionary with task names as keys and results as values
    """
    qwen_orig_folder = evals_dir / "Qwen3_orig"
    if not qwen_orig_folder.exists():
        print("Qwen3_orig folder not found")
        return {}
    
    # Find all JSON files in Qwen3_orig
    json_files = find_json_files(qwen_orig_folder)
    
    results = {}
    for json_file in json_files:
        task_results = extract_task_results(json_file)
        results.update(task_results)
    
    print(f"Processed Qwen3_orig: {len(results)} tasks")
    return results


def main():
    """Main function to process all models and create summary."""
    # Path to the evaluation logs
    evals_dir = Path("/home/dsi/lustigb/RL_for_small_LLM/open-rs/logs/evals")
    
    if not evals_dir.exists():
        print(f"Evaluation directory not found: {evals_dir}")
        return
    
    print(f"Processing evaluation results from: {evals_dir}")
    print("=" * 60)
    
    # Initialize the main results dictionary
    all_results = {}
    
    # Process Exp3Ben model (with iterations)
    print("Processing Exp3Ben model...")
    exp3ben_results = process_model_folders(evals_dir, "Exp3Ben")
    if exp3ben_results:
        all_results["Exp3Ben"] = exp3ben_results
        print(f"Found {len(exp3ben_results)} iterations for Exp3Ben")
    
    print()
    
    # Process Qwen3_trained model (with iterations)
    print("Processing Qwen3_trained model...")
    qwen3_trained_results = process_model_folders(evals_dir, "Qwen3_trained")
    if qwen3_trained_results:
        all_results["Qwen3_trained"] = qwen3_trained_results
        print(f"Found {len(qwen3_trained_results)} iterations for Qwen3_trained")
    
    print()
    
    # Process Qwen3_orig model (no iterations)
    print("Processing Qwen3_orig model...")
    qwen3_orig_results = process_qwen3_orig(evals_dir)
    if qwen3_orig_results:
        # Wrap in a single dictionary since there are no iterations
        all_results["Qwen3_orig"] = {0: qwen3_orig_results}  # Use 0 as the "iteration"
    
    print()
    print("=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    # Print summary of results
    for model_name, model_data in all_results.items():
        print(f"\n{model_name}:")
        if isinstance(model_data, dict):
            for iteration, tasks in model_data.items():
                if model_name == "Qwen3_orig":
                    print(f"  Tasks: {list(tasks.keys())}")
                else:
                    print(f"  Iteration {iteration}: {list(tasks.keys())}")
        
        # Show sample results for first iteration/task
        if model_data:
            first_iter = next(iter(model_data.values()))
            if first_iter:
                first_task_name = next(iter(first_iter.keys()))
                first_task_results = first_iter[first_task_name]
                print(f"    Sample results for '{first_task_name}': {first_task_results}")
    
    # Save results to JSON file
    output_file = Path("/home/dsi/lustigb/RL_for_small_LLM/open-rs/evaluation_results_summary.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return all_results


if __name__ == "__main__":
    results = main()