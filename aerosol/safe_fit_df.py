from aerosol.safe_fit_row import safe_fit_row

def safe_fit_df(df, max_retries=5, n_modes=None, n_samples=10000):
    """
    Safely fits aerosol modes for an entire DataFrame with retries and fallback strategies.

    Args:
        df: Input DataFrame (each row will be fitted separately)
        max_retries: Maximum number of retry attempts per row
        n_modes: Optional forced number of modes (None for auto-detection)
        n_samples: Number of samples for fitting (doubled on each retry)

    Returns:
        Tuple of (results_list, timings_list) where:
        - results_list: List of fitting results for each row
        - timings_list: List of elapsed times for each row
    """
    import numpy as np
    import pandas as pd
    import time

    safe_results = []
    safe_timings = []
    np.random.seed(42)
    for i in range(len(df)):
        start_time = time.time()

        try:
            # Get single row as DataFrame (preserve index)
            row_df = df.iloc[i:i + 1, :]

            # Perform the fit with retries
            fit_result = safe_fit_row(
                row_df,
                max_retries=max_retries,
                n_modes=n_modes,
                n_samples=n_samples,
            )

            elapsed_time = time.time() - start_time

            safe_results.append(fit_result)
            safe_timings.append(elapsed_time)

            print(f"Successfully fitted row {i + 1}/{len(df)} in {elapsed_time:.2f}s")

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"Failed to fit row {i + 1}/{len(df)} after {elapsed_time:.2f}s")
            print(f"Error: {str(e)}")

            # Append None for failed fits
            safe_results.append(None)
            safe_timings.append(elapsed_time)

    return safe_results, safe_timings
