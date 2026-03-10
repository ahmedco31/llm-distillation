# scripts/design_prompts.py (COMPLETE VERSION)
"""
Systematic prompt generator for reasoning analysis - COMPLETE IMPLEMENTATION
"""

import json
from typing import List, Dict
from datasets import load_dataset

class PromptDesigner:
    """Generates diverse reasoning prompts across multiple categories."""
    
    def __init__(self):
        self.categories = {
            'math_basic': {'count': 300, 'source': 'gsm8k'},
            'logic_chain': {'count': 300, 'source': 'generated'},
            'commonsense': {'count': 200, 'source': 'commonsense_qa'},
            'code_simple': {'count': 200, 'source': 'mbpp'},
            'abstract': {'count': 200, 'source': 'generated'}
        }
    
    def _estimate_difficulty(self, text: str) -> str:
        """Heuristic difficulty estimation."""
        word_count = len(text.split())
        has_multiple_steps = any(word in text.lower() 
                                for word in ['then', 'after', 'next', 'finally'])
        
        if word_count < 20 and not has_multiple_steps:
            return 'easy'
        elif word_count < 40 or has_multiple_steps:
            return 'medium'
        else:
            return 'hard'
    
    def load_math_prompts(self) -> List[Dict]:
        """Load GSM8K math problems."""
        print("Loading GSM8K dataset...")
        gsm8k = load_dataset("openai/gsm8k", "main", split="train")
        
        prompts = []
        for idx, item in enumerate(gsm8k.select(range(150, 450))):
            prompts.append({
                'id': f'math_v2_{idx:03d}',
                'prompt': item['question'],
                'category': 'math_basic',
                'difficulty': self._estimate_difficulty(item['question']),
                'source': 'gsm8k',
                'ground_truth': item['answer']
            })
        
        print(f"✓ Loaded {len(prompts)} math prompts")
        return prompts
    
    def load_commonsense_prompts(self) -> List[Dict]:
        """Load CommonsenseQA problems."""
        print("Loading CommonsenseQA dataset...")
        csqa = load_dataset("tau/commonsense_qa", split="train")
        
        prompts = []
        for idx, item in enumerate(csqa.select(range(100, 300))):
            # Format with choices
            choices_text = ", ".join(
                f"({label}) {text}" 
                for label, text in zip(item['choices']['label'], 
                                      item['choices']['text'])
            )
            full_prompt = f"{item['question']}\nOptions: {choices_text}"
            
            prompts.append({
                'id': f'common_v2_{idx:03d}',
                'prompt': full_prompt,
                'category': 'commonsense',
                'difficulty': 'medium',
                'source': 'commonsense_qa',
                'ground_truth': item['answerKey']
            })
        
        print(f"✓ Loaded {len(prompts)} commonsense prompts")
        return prompts
    
    def load_code_prompts(self) -> List[Dict]:
        """Load MBPP code problems."""
        print("Loading MBPP dataset...")
        try:
            mbpp = load_dataset("mbpp", split="train")
            
            prompts = []
            for idx, item in enumerate(mbpp.select(range(100, 300))):
                prompts.append({
                    'id': f'code_v2_{idx:03d}',
                    'prompt': item['text'],
                    'category': 'code_simple',
                    'difficulty': 'medium',
                    'source': 'mbpp',
                    'ground_truth': item['code']
                })
            
            print(f"✓ Loaded {len(prompts)} code prompts")
            return prompts
            
        except Exception as e:
            print(f"⚠ MBPP loading failed: {e}")
            print("  Generating code prompts from templates instead...")
            return self.generate_code_prompts(100)
    
    def generate_logic_prompts(self, count: int = 150) -> List[Dict]:
        """
        Generate logical reasoning prompts from templates.
        
        WHY: Limited high-quality logic datasets on HuggingFace.
        Template-based generation ensures deductive/inductive coverage.
        """
        templates = [
            # Syllogistic reasoning (easy)
            {
                'template': "If all {A} are {B}, and all {B} are {C}, what can we conclude about {A}?",
                'params': [
                    {'A': 'dogs', 'B': 'mammals', 'C': 'animals'},
                    {'A': 'roses', 'B': 'flowers', 'C': 'plants'},
                    {'A': 'squares', 'B': 'rectangles', 'C': 'shapes'},
                    {'A': 'teachers', 'B': 'educators', 'C': 'professionals'},
                    {'A': 'laptops', 'B': 'computers', 'C': 'electronic devices'},
                    {'A': 'nurses', 'B': 'healthcare workers', 'C': 'professionals'},
                    {'A': 'oak trees', 'B': 'trees', 'C': 'plants'},
                    {'A': 'Python', 'B': 'programming languages', 'C': 'tools'},
                    {'A': 'whales', 'B': 'mammals', 'C': 'living things'},
                    {'A': 'novels', 'B': 'books', 'C': 'written works'},
                ],
                'difficulty': 'easy'
            },
            # Two-premise reasoning (medium)
            {
                'template': "Premise 1: {premise1}\nPremise 2: {premise2}\nWhat logically follows?",
                'params': [
                    {'premise1': 'All engineers can code', 'premise2': 'Sarah is an engineer'},
                    {'premise1': 'No reptiles have fur', 'premise2': 'A snake is a reptile'},
                    {'premise1': 'Some students study hard', 'premise2': 'All who study hard succeed'},
                    {'premise1': 'Every mammal has a spine', 'premise2': 'Dolphins are mammals'},
                    {'premise1': 'No metal conducts heat poorly', 'premise2': 'Copper is a metal'},
                    {'premise1': 'All birds have wings', 'premise2': 'A penguin is a bird'},
                    {'premise1': 'No fish can survive on land', 'premise2': 'A salmon is a fish'},
                    {'premise1': 'All planets orbit a star', 'premise2': 'Earth is a planet'},
                    {'premise1': 'Some languages are tonal', 'premise2': 'Mandarin is a tonal language'},
                    {'premise1': 'No prime number is even except 2', 'premise2': '7 is a prime number'},
                ],
                'difficulty': 'medium'
            },
            # Conditional reasoning (medium)
            {
                'template': "If {condition}, then {consequence}. We observe that {observation}. What can we infer?",
                'params': [
                    {'condition': 'it rains', 'consequence': 'the ground gets wet', 
                     'observation': 'the ground is wet'},
                    {'condition': 'a number is divisible by 6', 'consequence': 'it is divisible by 3', 
                     'observation': 'a number is divisible by 6'},
                    {'condition': 'someone is a doctor', 'consequence': 'they have a medical degree', 
                     'observation': 'John is a doctor'},
                    {'condition': 'the alarm goes off', 'consequence': 'someone entered the building', 
                     'observation': 'the alarm is ringing'},
                    {'condition': 'water freezes', 'consequence': 'the temperature is below 0°C', 
                     'observation': 'the water has frozen'},
                    {'condition': 'the power goes out', 'consequence': 'the lights turn off', 'observation': 'the lights are off'},
                    {'condition': 'a shape has 4 equal sides', 'consequence': 'it is a rhombus', 'observation': 'a shape has 4 equal sides'},
                    {'condition': 'someone exercises daily', 'consequence': 'they improve their fitness', 'observation': 'Maria exercises daily'},
                    {'condition': 'a file is deleted', 'consequence': 'it moves to the trash', 'observation': 'the file was deleted'},
                    {'condition': 'it snows heavily', 'consequence': 'roads become slippery', 'observation': 'it snowed heavily last night'},
                ],
                'difficulty': 'medium'
            },
            # Negation reasoning (hard)
            {
                'template': "Given: Not all {group} are {property}. Also given: {specific} is a {group}. Can we conclude that {specific} is {property}?",
                'params': [
                    {'group': 'birds', 'property': 'able to fly', 'specific': 'a penguin'},
                    {'group': 'metals', 'property': 'magnetic', 'specific': 'aluminum'},
                    {'group': 'fruits', 'property': 'sweet', 'specific': 'a lemon'},
                    {'group': 'students', 'property': 'good at math', 'specific': 'John'},
                    {'group': 'politicians', 'property': 'honest', 'specific': 'the senator'},
                    {'group': 'doctors', 'property': 'specialized in surgery', 'specific': 'Dr. Smith'},
                    {'group': 'cars', 'property': 'electric', 'specific': 'a Toyota Corolla'},
                    {'group': 'languages', 'property': 'written left to right', 'specific': 'Arabic'},
                    {'group': 'athletes', 'property': 'professional', 'specific': 'James'},
                    {'group': 'vegetables', 'property': 'green', 'specific': 'a carrot'},
                ],
                'difficulty': 'hard'
            },
            # Contradiction detection (hard)
            {
                'template': "Statement A: {statement_a}\nStatement B: {statement_b}\nAre these statements contradictory, compatible, or independent?",
                'params': [
                    {'statement_a': 'All cats are animals', 'statement_b': 'Some animals are not cats'},
                    {'statement_a': 'The meeting is at 3 PM', 'statement_b': 'The meeting is not at 2 PM'},
                    {'statement_a': 'Every student passed', 'statement_b': 'At least one student failed'},
                    {'statement_a': 'The box is red', 'statement_b': 'The box is blue'},
                    {'statement_a': 'It will rain tomorrow', 'statement_b': 'The weather will be sunny'},
                    {'statement_a': 'All swans are white', 'statement_b': 'Some swans are black'},
                    {'statement_a': 'The file was saved', 'statement_b': 'No changes were written to disk'},
                    {'statement_a': 'The train arrives at noon', 'statement_b': 'The train does not arrive before 1 PM'},
                    {'statement_a': 'All numbers are positive', 'statement_b': 'Zero is a number'},
                    {'statement_a': 'She passed every exam', 'statement_b': 'She failed her biology exam'},
                ],
                'difficulty': 'hard'
            },
        ]
        
        prompts = []
        template_idx = 0
        
        # Cycle through templates until we reach count
        while len(prompts) < count:
            template_data = templates[template_idx % len(templates)]
            template = template_data['template']
            params_list = template_data['params']
            
            for params in params_list:
                if len(prompts) >= count:
                    break
                
                prompt_text = template.format(**params)
                prompts.append({
                    'id': f'logic_{len(prompts):03d}',
                    'prompt': prompt_text,
                    'category': 'logic_chain',
                    'difficulty': template_data['difficulty'],
                    'source': 'template_generated',
                    'ground_truth': None
                })
            
            template_idx += 1
        
        print(f"✓ Generated {len(prompts)} logic prompts")
        return prompts[:count]  # Trim to exact count
    
    def generate_code_prompts(self, count: int = 100) -> List[Dict]:
        """
        Generate code reasoning prompts from templates.
        
        WHY: Fallback if MBPP fails to load. Tests algorithmic thinking.
        """
        templates = [
            "Write a Python function that takes a list of numbers and returns the sum of all even numbers.",
            "Create a function that checks if a string is a palindrome (reads the same forwards and backwards).",
            "Write a function that finds the largest number in a list without using the max() function.",
            "Implement a function that reverses a string without using built-in reverse methods.",
            "Write a function that counts how many times each character appears in a string.",
            "Create a function that removes all duplicates from a list while preserving order.",
            "Write a function that merges two sorted lists into one sorted list.",
            "Implement a function that checks if a number is prime.",
            "Write a function that calculates the factorial of a number recursively.",
            "Create a function that finds the second largest number in a list.",
            "Write a function that converts a decimal number to binary.",
            "Implement a function that finds all pairs of numbers in a list that sum to a target value.",
            "Write a function that rotates a list by n positions to the right.",
            "Create a function that checks if two strings are anagrams of each other.",
            "Write a function that flattens a nested list into a single-level list.",
            "Implement a function that finds the missing number in a sequence from 1 to n.",
            "Write a function that generates the first n Fibonacci numbers.",
            "Create a function that capitalizes the first letter of each word in a sentence.",
            "Write a function that removes all vowels from a string.",
            "Implement a function that checks if a list is sorted in ascending order.",
        ]
        
        prompts = []
        while len(prompts) < count:
            template = templates[len(prompts) % len(templates)]
            
            prompts.append({
                'id': f'code_{len(prompts):03d}',
                'prompt': template,
                'category': 'code_simple',
                'difficulty': 'medium',
                'source': 'template_generated',
                'ground_truth': None
            })
        
        print(f"✓ Generated {len(prompts)} code prompts")
        return prompts[:count]
    
    def generate_abstract_prompts(self, count: int = 100) -> List[Dict]:
        """
        Generate abstract reasoning prompts.
        
        WHY: Pattern recognition tests transfer learning - key indicator
        of true understanding vs. memorization.
        """
        templates = [
            # Analogies
            {
                'template': "{a1} is to {a2} as {b1} is to what?",
                'params': [
                    {'a1': 'hot', 'a2': 'cold', 'b1': 'tall'},
                    {'a1': 'dog', 'a2': 'puppy', 'b1': 'cat'},
                    {'a1': 'hand', 'a2': 'glove', 'b1': 'foot'},
                    {'a1': 'teacher', 'a2': 'student', 'b1': 'doctor'},
                    {'a1': 'day', 'a2': 'night', 'b1': 'summer'},
                    {'a1': 'book', 'a2': 'read', 'b1': 'music'},
                    {'a1': 'up', 'a2': 'down', 'b1': 'left'},
                    {'a1': 'happy', 'a2': 'sad', 'b1': 'love'},
                    {'a1': 'pen', 'a2': 'write', 'b1': 'knife'},
                    {'a1': 'sun', 'a2': 'solar system', 'b1': 'Earth'},
                    {'a1': 'flour', 'a2': 'bread', 'b1': 'grapes'},
                    {'a1': 'fast', 'a2': 'slow', 'b1': 'loud'},
                    {'a1': 'chapter', 'a2': 'book', 'b1': 'verse'},
                ],
                'difficulty': 'medium'
            },
            # Number sequences
            {
                'template': "What comes next in this sequence: {sequence}",
                'params': [
                    {'sequence': '2, 4, 8, 16, __'},
                    {'sequence': '1, 1, 2, 3, 5, 8, __'},
                    {'sequence': '100, 50, 25, 12.5, __'},
                    {'sequence': '3, 6, 9, 12, __'},
                    {'sequence': '1, 4, 9, 16, 25, __'},
                    {'sequence': '2, 6, 12, 20, 30, __'},
                    {'sequence': '5, 10, 20, 40, __'},
                    {'sequence': '1, 2, 4, 7, 11, __'},
                    {'sequence': '0, 1, 3, 6, 10, __'},
                    {'sequence': '81, 27, 9, 3, __'},
                    {'sequence': '1, 8, 27, 64, __'},
                    {'sequence': '10, 9, 7, 4, __'},
                ],
                'difficulty': 'medium'
            },
            # Letter sequences
            {
                'template': "Complete the pattern: {sequence}",
                'params': [
                    {'sequence': 'A, C, E, G, __'},
                    {'sequence': 'Z, Y, X, W, __'},
                    {'sequence': 'B, D, F, H, __'},
                    {'sequence': 'AZ, BY, CX, DW, __'},
                    {'sequence': 'A, E, I, M, __'},
                    {'sequence': 'AA, BB, CC, DD, __'},
                    {'sequence': 'ZA, YB, XC, WD, __'},
                    {'sequence': 'A, Z, B, Y, C, __'},
                ],
                'difficulty': 'medium'
            },
            # Pattern explanation
            {
                'template': "Identify the underlying pattern in this series and explain it: {series}",
                'params': [
                    {'series': '1, 3, 6, 10, 15, 21'},
                    {'series': '2, 5, 11, 23, 47'},
                    {'series': 'Monday, Wednesday, Friday, __'},
                    {'series': 'January, March, May, July, __'},
                    {'series': '0, 1, 4, 9, 16, 25'},
                    {'series': '1, 2, 6, 24, 120'},
                    {'series': 'Sunday, Tuesday, Thursday, __'},
                    {'series': 'February, April, June, August, __'},
                ],
                'difficulty': 'hard'
            },
            # Odd one out
            {
                'template': "Which item does not belong in this group and why? {items}",
                'params': [
                    {'items': 'apple, orange, banana, carrot'},
                    {'items': 'car, bus, train, bicycle, airplane'},
                    {'items': '2, 4, 6, 8, 9, 10'},
                    {'items': 'red, green, blue, heavy'},
                    {'items': 'square, circle, triangle, cube'},
                    {'items': 'piano, guitar, violin, microphone'},
                    {'items': '3, 5, 7, 9, 11'},
                    {'items': 'Paris, London, Tokyo, Amazon'},
                    {'items': 'walking, running, swimming, sleeping'},
                    {'items': 'circle, oval, sphere, ellipse'},
                ],
                'difficulty': 'medium'
            },
        ]
        
        prompts = []
        while len(prompts) < count:
            template_data = templates[len(prompts) % len(templates)]
            template = template_data['template']
            params_list = template_data['params']
            param_idx = (len(prompts) // len(templates)) % len(params_list)
            params = params_list[param_idx]
            
            prompt_text = template.format(**params)
            prompts.append({
                'id': f'abstract_{len(prompts):03d}',
                'prompt': prompt_text,
                'category': 'abstract',
                'difficulty': template_data['difficulty'],
                'source': 'template_generated',
                'ground_truth': None
            })
        
        print(f"✓ Generated {len(prompts)} abstract prompts")
        return prompts[:count]
    
    def export_prompt_dataset(self, output_path: str):
        """Export complete 600-prompt dataset to JSON."""
        print("=" * 50)
        print("BUILDING COMPLETE PROMPT DATASET")
        print("=" * 50)
        
        # Load/generate all categories
        all_prompts = []
        all_prompts.extend(self.load_math_prompts())           # 300
        all_prompts.extend(self.load_commonsense_prompts())    # 200
        all_prompts.extend(self.load_code_prompts())           # 200
        all_prompts.extend(self.generate_logic_prompts(300))   # 300
        all_prompts.extend(self.generate_abstract_prompts(200)) # 200
        
        # Verify counts
        from collections import Counter
        category_counts = Counter(p['category'] for p in all_prompts)
        
        print("\n" + "=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)
        print(f"Total prompts: {len(all_prompts)}")
        print("\nBy category:")
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat}: {count}")
        
        # Export
        with open(output_path, 'w') as f:
            json.dump(all_prompts, f, indent=2)
        
        print(f"\n✓ Exported to {output_path}")
        
        # Also export as CSV for convenience
        import pandas as pd
        df = pd.DataFrame(all_prompts)
        csv_path = output_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Also exported to {csv_path}")


# Run this script
if __name__ == "__main__":
    from pathlib import Path
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    
    designer = PromptDesigner()
    designer.export_prompt_dataset('data/prompts_v2.json')
    
    print("\n" + "=" * 50)
    print("✅ COMPLETE: 1200 prompts ready for collection!")
    print("=" * 50)
