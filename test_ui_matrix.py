#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick test for UI improvements"""

import sys
import pandas as pd
from logic.conversion import generate_catalyst_pathway_matrix

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("Testing UI improvements for matrix display")
print("=" * 70)

try:
    # Generate matrix
    matrix_df = generate_catalyst_pathway_matrix()
    print("\nGenerated matrix:")
    print(matrix_df.to_string())

    # Test plain table formatting
    print("\n" + "=" * 70)
    print("Plain table format:")
    plain_df = matrix_df.copy()
    for col in plain_df.columns:
        if col != 'Pathway':
            plain_df[col] = plain_df[col].apply(lambda x: 'N/A' if pd.isna(x) else f'{x:.1f}%')
    print(plain_df.to_string())

    # Test statistics
    print("\n" + "=" * 70)
    print("Statistics:")

    # Find best combination
    max_val = -1
    best_pathway = ""
    best_catalyst = ""
    for col in matrix_df.columns:
        if col != 'Pathway':
            col_max = matrix_df[col].max()
            if not pd.isna(col_max) and col_max > max_val:
                max_val = col_max
                best_catalyst = col
                best_pathway = matrix_df[matrix_df[col] == col_max]['Pathway'].values[0]

    print(f"Best combination: {best_pathway} + {best_catalyst} = {max_val:.1f}%")

    # Average yield
    all_vals = []
    for col in matrix_df.columns:
        if col != 'Pathway':
            all_vals.extend(matrix_df[col].dropna().tolist())
    avg_yield = sum(all_vals) / len(all_vals) if all_vals else 0
    print(f"Average yield: {avg_yield:.1f}%")

    # Valid combinations
    valid_count = sum([len(matrix_df[col].dropna()) for col in matrix_df.columns if col != 'Pathway'])
    total_count = len(matrix_df) * (len(matrix_df.columns) - 1)
    print(f"Valid combinations: {valid_count}/{total_count}")

    print("\nTest PASSED!")

except Exception as e:
    print(f"Test FAILED: {e}")
    import traceback
    traceback.print_exc()
