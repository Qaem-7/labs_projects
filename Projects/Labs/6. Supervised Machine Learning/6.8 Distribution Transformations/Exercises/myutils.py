import pandas as pd
import numpy as np

def skew_calc(df):
    """
    Diagnoses skewness for every numeric column in a DataFrame and recommends a transformation based on the column's skewness and
    minimum value. Binary, encoded, and ID columns are excluded, since skewness isn't a meaningful for them.
    It returns a DataFrame with the following columns:
    Feature, Skewness, Degree, Direction, Recommended Transformation
    """
    # Your code here
    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        skew_val = df[col].skew()
        abs_skew = abs(skew_val)
        min_val = df[col].min()

        # Classification
        if abs_skew < 0.5:
            degree = 'Approximately Symmetric'
        elif abs_skew <= 1.0:
            degree = 'Moderately Skewed'
        else:
            degree = 'Highly Skewed'

        direction = 'Positive' if skew_val >= 0 else 'Negative'

        # Recommendation
        if degree == 'Approximately Symmetric':
            recommend = 'None needed'
        elif min_val >= 0:
            recommend = 'log(x+1) or Yeo-Johnson' if min_val == 0 else 'Box-Cox or Yeo-Johnson'
        else:
            recommend = 'Yeo-Johnson'

        results.append({
            'Feature': col,
            'Skewness': skew_val,
            'Degree': degree,
            'Direction': direction,
            'Recommended Transformation': recommend
        })

    return pd.DataFrame(results)
