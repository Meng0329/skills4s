# Systematic Debugging Skill

Follow this procedure:

1. REPRODUCE
   Run the relevant tests and establish the failure.

2. INSPECT_TEST
   Read the failing test and identify the expected behavior.

3. INSPECT_IMPLEMENTATION
   Read the relevant implementation and locate the discrepancy.

4. REPAIR
   Make the smallest change that addresses the root cause.

5. TARGET_VERIFY
   Run the directly affected test.

6. REGRESSION_VERIFY
   Run the complete test suite before finishing.
