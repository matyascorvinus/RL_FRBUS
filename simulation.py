# Before the solver call
logger.debug(f"Input values before solve: {solutions}")
logger.debug(f"Quarter being processed: {quarter_str}")

# Monitor specific variables that might cause issues
if np.any(solutions < 0):
    logger.warning(f"Negative values detected in solutions: {solutions[solutions < 0]}")