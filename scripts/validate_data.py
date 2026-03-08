# scripts/validate_data.py
"""
Data quality validation.

"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class DataValidator:
    """Validates collected data quality and generates QA report."""
    
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.issues = []
    
    def run_validation(self) -> Dict:
        """Run full validation suite."""
        checks = {
            'completeness': self._check_completeness(),
            'duplicates': self._check_duplicates(),
            'response_quality': self._check_response_quality(),
            'metadata_integrity': self._check_metadata(),
            'category_distribution': self._check_distribution()
        }
        
        return checks
    
    def _check_completeness(self) -> Dict:
        """Check for missing values."""
        missing = self.df.isnull().sum()
        critical_columns = ['prompt', 'response', 'category']
        
        has_missing_critical = any(missing[col] > 0 for col in critical_columns)
        
        return {
            'status': 'PASS' if not has_missing_critical else 'FAIL',
            'missing_counts': missing.to_dict(),
            'completeness_rate': (1 - self.df.isnull().sum().sum() / self.df.size) * 100
        }
    
    def _check_duplicates(self) -> Dict:
        """Detect duplicate prompts or responses."""
        prompt_dups = self.df['prompt'].duplicated().sum()
        response_dups = self.df['response'].duplicated().sum()
        
        return {
            'status': 'PASS' if prompt_dups == 0 else 'WARNING',
            'duplicate_prompts': int(prompt_dups),
            'duplicate_responses': int(response_dups),
            'note': 'Duplicate responses may indicate model memorization'
        }
    
    def _check_response_quality(self) -> Dict:
        """Validate response characteristics."""
        self.df['response_length'] = self.df['response'].str.len()
        
        return {
            'status': 'PASS',
            'mean_length': int(self.df['response_length'].mean()),
            'min_length': int(self.df['response_length'].min()),
            'max_length': int(self.df['response_length'].max()),
            'short_responses': int((self.df['response_length'] < 50).sum()),
            'note': 'Short responses (<50 chars) may indicate errors'
        }
    
    def _check_metadata(self) -> Dict:
        """Verify metadata consistency."""
        required_fields = ['category', 'model', 'timestamp']
        missing_fields = [f for f in required_fields if f not in self.df.columns]
        
        return {
            'status': 'PASS' if not missing_fields else 'FAIL',
            'missing_fields': missing_fields,
            'timestamp_range': {
                'start': str(self.df['timestamp'].min()),
                'end': str(self.df['timestamp'].max())
            }
        }
    
    def _check_distribution(self) -> Dict:
        """Analyze category distribution."""
        dist = self.df['category'].value_counts()
        
        return {
            'status': 'PASS',
            'distribution': dist.to_dict(),
            'balance_score': float(dist.std() / dist.mean()),  # Lower = more balanced
            'note': 'Balance score < 0.5 indicates good distribution'
        }
    
    def generate_report(self, output_path: str):
        """Generate human-readable validation report."""
        results = self.run_validation()
        
        report = f"""
# DATA VALIDATION REPORT
Generated: {pd.Timestamp.now()}
Dataset: {len(self.df)} samples

## Summary
{self._format_results(results)}

## Recommendations
{self._generate_recommendations(results)}
"""
        
        Path(output_path).write_text(report)
        print(f"✓ Validation report saved to {output_path}")
        
        return results
    
    def _format_results(self, results: Dict) -> str:
        """Format results as markdown."""
        lines = []
        for check, result in results.items():
            status = result.get('status', 'UNKNOWN')
            emoji = '✓' if status == 'PASS' else ('⚠' if status == 'WARNING' else '✗')
            lines.append(f"{emoji} **{check.upper()}**: {status}")
        return '\n'.join(lines)
    
    def _generate_recommendations(self, results: Dict) -> str:
        """Generate actionable recommendations."""
        recs = []
        
        if results['completeness']['status'] != 'PASS':
            recs.append("- Remove rows with missing critical fields")
        
        if results['duplicates']['duplicate_prompts'] > 10:
            recs.append("- Investigate duplicate prompts - may affect analysis")
        
        if results['response_quality']['short_responses'] > 50:
            recs.append("- Review short responses for API errors")
        
        if results['category_distribution']['balance_score'] > 0.5:
            recs.append("- Consider rebalancing categories for fair analysis")
        
        return '\n'.join(recs) if recs else "No issues detected - dataset ready for analysis!"

if __name__ == "__main__":
    validator = DataValidator('data/responses_openai.csv')
    validator.generate_report('data/validation_report.md')
