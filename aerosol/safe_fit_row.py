
def safe_fit_row(df_row, max_retries=5, n_modes=None, n_samples=10000, fallback_n_modes=2):
    """
    Safely fits aerosol modes with robust retries and fallback strategies.

    Args:
        df_row: Input data (DataFrame, Series, or dict)
        max_retries: Maximum retry attempts (default: 5)
        n_modes: Optional forced number of modes
        n_samples: Initial samples for fitting (default: 10000)
        fallback_n_modes: Fallback mode count when detection fails (default: 2)

    Returns:
        Tuple of (results, timing) from successful fit

    Raises:
        ValueError: For invalid inputs
        RuntimeError: If all retries fail
    """
    import pandas as pd
    import numpy as np
    import time
    import aerosol.fitting as af
    from aerosol.fitting import fit_multimodes
    # Input conversion and validation
    try:
        if isinstance(df_row, (pd.Series, dict)):
            df_row = pd.DataFrame([df_row.values()] if isinstance(df_row, dict) else [df_row.values],
                                  columns=df_row.keys() if isinstance(df_row, dict) else df_row.index)

        df_row = df_row.astype(float)  # Ensure numeric data
        if df_row.empty or df_row.isna().all().all():
            raise ValueError("Input contains no valid data")

    except Exception as e:
        raise ValueError(f"Invalid input data: {str(e)}") from e
    # np.random.seed(42)
    # Adaptive retry configuration
    retry_delay = 0.1  # Initial delay in seconds
    max_delay = 2.0  # Maximum delay between retries

    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            result = fit_multimodes(df_row, n_modes=n_modes, n_samples=n_samples)

            # Check for specific failure modes
            if result is None or len(result[0]) == 0:
                raise RuntimeError("Empty result returned")

            if result[0][0].get('number_of_gaussians', 0) == 0:
                raise RuntimeError("No modes detected")

            return result

        except Exception as e:
            last_exception = e
            error_str = str(e).lower()

            if "lsq failed" in error_str:
                print(f"Attempt {attempt}: LSQ failed - doubling samples to {n_samples * 2}")
                n_samples *= 2
            elif "memory" in error_str:
                print(f"Attempt {attempt}: Memory issue - reducing samples by 25%")
                n_samples = int(n_samples * 0.75)
            elif "kneedle" in error_str:
                print(f"Attempt {attempt}: Mode detection failed - using fallback {fallback_n_modes} modes")
                return fit_multimodes(df_row, n_modes=fallback_n_modes)

            # Adaptive backoff
            time.sleep(min(retry_delay * (1.5 ** attempt), max_delay))

    # Final fallback attempt
    print(f"All attempts failed - final fallback to {fallback_n_modes} modes")
    return fit_multimodes(df_row, n_modes=fallback_n_modes)
