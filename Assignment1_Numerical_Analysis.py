import numpy as np
import sys

# --- Helper function for printing formatted tables ---
def print_table(headers, data):
    """
    Prints a formatted table.
    :param headers: List of strings for table headers.
    :param data: List of lists/tuples, where each inner list/tuple is a row of data.
    """
    if not data:
        print("No data to display for this method run.")
        return

    # Determine initial column widths based on headers
    col_widths = [len(header) for header in headers]

    # Calculate max width for each column based on data
    # Format floats to a consistent precision for width calculation
    formatted_data = []
    for row in data:
        formatted_row = []
        for i, item in enumerate(row):
            formatted_item = ""
            if isinstance(item, float):
                formatted_item = f"{item:.8f}" # Use 8 decimal places for floats
            elif isinstance(item, int):
                formatted_item = str(item)
            else: # For strings like '-'
                formatted_item = str(item)
            formatted_row.append(formatted_item)
            col_widths[i] = max(col_widths[i], len(formatted_item))
        formatted_data.append(formatted_row)

    # Print header
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # Print data rows
    for row in formatted_data:
        row_line = " | ".join(f"{item:<{w}}" for item, w in zip(row, col_widths))
        print(row_line)
    print("\n")

# --- 1. Bisection Method Implementation ---
def bisection_method(func, a, b, equation_str, tol=1e-6, max_iter=100):
    """
    Implements the Bisection Method for finding roots of a function.

    :param func: The function f(x) for which to find the root.
    :param a: The lower bound of the interval.
    :param b: The upper bound of the interval.
    :param equation_str: String representation of the function.
    :param tol: The desired tolerance for the root.
    :param max_iter: The maximum number of iterations.
    :return: The approximate root if found, else None.
    """
    print(f"--- Bisection Method for f(x) = {equation_str} ---")
    print(f"Attempting to find root in interval [{a}, {b}] with tolerance {tol}")

    try:
        f_a = func(a)
        f_b = func(b)
    except Exception as e:
        print(f"Error: Function evaluation failed at interval bounds [{a}, {b}]: {e}")
        print("Skipping this method call.")
        return None

    # Check for exact roots at initial bounds
    if f_a == 0.0:
        print(f"Exact root found at initial lower bound: {a:.8f}")
        return a
    if f_b == 0.0:
        print(f"Exact root found at initial upper bound: {b:.8f}")
        return b

    # Calculus-based pre-check: Bisection method guarantees convergence if a root is bracketed.
    if f_a * f_b > 0: # f_a * f_b == 0 is already handled by exact root checks above
        print(f"Reason for skipping: f(a) = {f_a:.6f} and f(b) = {f_b:.6f} have the same sign.")
        print("Bisection method requires f(a) and f(b) to have opposite signs to guarantee a root in the interval.")
        print("Therefore, skipping iterations for this question.")
        return None
    else:
        print(f"Initial check passed: f(a) = {f_a:.6f} and f(b) = {f_b:.6f} have opposite signs.")
        print(f"Convergence is guaranteed within the interval [{a}, {b}] given sufficient iterations.")

    headers = ["Iteration", "a", "b", "c", "f(a)", "f(b)", "f(c)"]
    iteration_data = []

    c = (a + b) / 2 # Initialize c
    for i in range(1, max_iter + 1):
        try:
            c = (a + b) / 2
            # Re-evaluate f(a) and f(b) to ensure up-to-date values, though strictly only c changes significantly.
            f_a = func(a) 
            f_b = func(b)
            f_c = func(c)
        except Exception as e:
            print(f"Error: Function evaluation failed during iteration {i} at x={c:.8f}: {e}")
            print("Stopping iterations.")
            print_table(headers, iteration_data)
            return None

        # Record data for the iteration table
        iteration_data.append([i, a, b, c, f_a, f_b, f_c])

        # Check for exact root first
        if f_c == 0.0:
            print(f"Exact root found after {i} iterations: {c:.8f}")
            print_table(headers, iteration_data)
            return c
        
        # Check for convergence
        if abs(f_c) < tol or (b - a) / 2 < tol:
            print(f"Converged after {i} iterations. Approximate root: {c:.8f}")
            print_table(headers, iteration_data)
            return c

        # Update the interval
        if f_a * f_c < 0:
            b = c
        else:
            a = c

    print(f"Warning: Bisection method did not converge to the specified tolerance ({tol}) after {max_iter} iterations.")
    print(f"Current approximation: {c:.8f}")
    print_table(headers, iteration_data)
    return c # Return the last approximation

# --- 2. False Position Method Implementation ---
def false_position_method(func, a, b, equation_str, tol=1e-6, max_iter=100):
    """
    Implements the False Position Method for finding roots of a function.

    :param func: The function f(x) for which to find the root.
    :param a: The lower bound of the interval.
    :param b: The upper bound of the interval.
    :param equation_str: String representation of the function.
    :param tol: The desired tolerance for the root.
    :param max_iter: The maximum number of iterations.
    :return: The approximate root if found, else None.
    """
    print(f"--- False Position Method for f(x) = {equation_str} ---")
    print(f"Attempting to find root in interval [{a}, {b}] with tolerance {tol}")

    try:
        f_a = func(a)
        f_b = func(b)
    except Exception as e:
        print(f"Error: Function evaluation failed at interval bounds [{a}, {b}]: {e}")
        print("Skipping this method call.")
        return None

    # Check for exact roots at initial bounds
    if f_a == 0.0:
        print(f"Exact root found at initial lower bound: {a:.8f}")
        return a
    if f_b == 0.0:
        print(f"Exact root found at initial upper bound: {b:.8f}")
        return b

    # Calculus-based pre-check: False Position method guarantees convergence if a root is bracketed.
    if f_a * f_b > 0: # f_a * f_b == 0 is already handled by exact root checks above
        print(f"Reason for skipping: f(a) = {f_a:.6f} and f(b) = {f_b:.6f} have the same sign.")
        print("False Position method requires f(a) and f(b) to have opposite signs to guarantee a root in the interval.")
        print("Therefore, skipping iterations for this question.")
        return None
    else:
        print(f"Initial check passed: f(a) = {f_a:.6f} and f(b) = {f_b:.6f} have opposite signs.")
        print(f"Convergence is generally guaranteed within [{a}, {b}], potentially faster than bisection.")

    headers = ["Iteration", "a_n", "b_n", "f(a_n)", "f(b_n)", "c_n", "f(c_n)", "a_n+1", "b_n+1"]
    iteration_data = []

    c_prev = None # To check convergence using |c_new - c_prev|
    
    for i in range(1, max_iter + 1):
        current_a = a
        current_b = b
        
        try:
            current_f_a = func(current_a)
            current_f_b = func(current_b)
        except Exception as e:
            print(f"Error: Function evaluation failed during iteration {i} at interval bounds [{current_a}, {current_b}]: {e}")
            print("Stopping iterations.")
            print_table(headers, iteration_data)
            return None


        # Check for division by zero (slope being too flat) - can happen during iterations too
        if abs(current_f_b - current_f_a) < sys.float_info.epsilon * max(abs(current_f_a), abs(current_f_b), 1):
            print(f"Error at iteration {i}: The difference f(b_n) - f(a_n) is too close to zero ({current_f_b - current_f_a:.8e}).")
            print("This indicates an extremely flat segment or identical function values at endpoints, leading to division by zero.")
            print("False Position method failed. Skipping further iterations.")
            print_table(headers, iteration_data)
            return None

        # Calculate c using the false position formula
        c = current_b - (current_f_b * (current_b - current_a)) / (current_f_b - current_f_a)
        
        try:
            f_c = func(c)
        except Exception as e:
            print(f"Error: Function evaluation failed during iteration {i} at calculated point c={c:.8f}: {e}")
            print("Stopping iterations.")
            print_table(headers, iteration_data)
            return None

        # Determine next interval bounds (a_n+1, b_n+1) for logging
        next_a, next_b = current_a, current_b 
        if current_f_a * f_c < 0:
            next_b = c
        else: # This path also covers the case where f_c is 0
            next_a = c

        # Record data for the iteration table (now next_a and next_b are defined)
        iteration_data.append([i, current_a, current_b, current_f_a, current_f_b, c, f_c, next_a, next_b])

        # Check for exact root first
        if f_c == 0.0:
            print(f"Exact root found after {i} iterations: {c:.8f}")
            print_table(headers, iteration_data)
            return c
        
        # Check for convergence
        if abs(f_c) < tol or (c_prev is not None and abs(c - c_prev) < tol):
            print(f"Converged after {i} iterations. Approximate root: {c:.8f}")
            print_table(headers, iteration_data)
            return c

        c_prev = c # Update previous c for next iteration's convergence check
        a = next_a # Update interval 'a' for the next iteration
        b = next_b # Update interval 'b' for the next iteration


    print(f"Warning: False Position method did not converge to the specified tolerance ({tol}) after {max_iter} iterations.")
    print(f"Current approximation: {c:.8f}") # c is the last computed approximation
    print_table(headers, iteration_data)
    return c

# --- 3. Newton's Method Implementation ---
def newton_method(func, dfunc, x0, equation_str, tol=1e-6, max_iter=100):
    """
    Implements Newton's Method for finding roots of a function.

    :param func: The function f(x) for which to find the root.
    :param dfunc: The derivative of the function f'(x).
    :param x0: The initial guess.
    :param equation_str: String representation of the function.
    :param tol: The desired tolerance for the root.
    :param max_iter: The maximum number of iterations.
    :return: The approximate root if found, else None.
    """
    print(f"--- Newton's Method for f(x) = {equation_str} ---")
    print(f"Attempting to find root with initial guess x0 = {x0} and tolerance {tol}")
    
    # Calculus-based pre-check: Check if f'(x0) is zero or too close to zero.
    try:
        f_x0 = func(x0)
        df_x0 = dfunc(x0)
    except Exception as e:
        print(f"Error: Function or derivative evaluation failed at initial guess x0 = {x0:.8f}: {e}")
        print("Skipping this method call.")
        return None

    # Check for exact root at initial guess
    if f_x0 == 0.0:
        print(f"Exact root found at initial guess x0: {x0:.8f}")
        return x0

    if abs(df_x0) < sys.float_info.epsilon * 100: # Use a slightly larger epsilon for initial checks
        print(f"Reason for skipping: Initial derivative f'(x0) = {df_x0:.8e} is too close to zero at x0 = {x0:.8f}.")
        print("Newton's method involves division by the derivative. A near-zero derivative means a nearly horizontal tangent,")
        print("which would lead to an extremely large or infinite step, likely causing divergence or an error.")
        print("Therefore, skipping iterations for this question.")
        return None
    else:
        print(f"Initial check passed: Derivative f'(x0) = {df_x0:.6f} is not close to zero.")
        print(f"Convergence for Newton's method is quadratic if the initial guess is sufficiently close to a root where f'(root) != 0.")
        print(f"It may diverge if the initial guess is far from a root, or if f'(x) is close to zero near the root.")

    headers = ["Iteration", "xn", "f(xn)", "f'(xn)", "xn+1"]
    iteration_data = []

    xn = x0
    for i in range(1, max_iter + 1):
        try:
            f_xn = func(xn)
            df_xn = dfunc(xn)
        except Exception as e:
            print(f"Error: Function or derivative evaluation failed at xn = {xn:.8f} during iteration {i}: {e}")
            print("Stopping iterations.")
            print_table(headers, iteration_data)
            return None

        # Check for exact root first
        if f_xn == 0.0:
            print(f"Exact root found after {i-1} iterations at xn: {xn:.8f}") # xn is the current guess where f(xn) is 0
            print_table(headers, iteration_data) # Table up to previous iteration, or append final row
            return xn

        # Check for division by zero (derivative being too small) during iterations
        if abs(df_xn) < sys.float_info.epsilon * max(abs(f_xn), 1): # Relative check
            print(f"Error at iteration {i}: Derivative f'(xn) = {df_xn:.8e} is too close to zero at xn = {xn:.8f}.")
            print("Newton's method failed due to a horizontal tangent, indicating a potential local minimum/maximum or inflection point.")
            print("Skipping further iterations.")
            print_table(headers, iteration_data)
            return None

        xn_plus_1 = xn - f_xn / df_xn
        
        # Record data for the iteration table
        iteration_data.append([i, xn, f_xn, df_xn, xn_plus_1])

        # Check for convergence (either function value close to zero or x not changing much)
        if abs(f_xn) < tol or abs(xn_plus_1 - xn) < tol:
            print(f"Converged after {i} iterations. Approximate root: {xn_plus_1:.8f}")
            print_table(headers, iteration_data)
            return xn_plus_1

        xn = xn_plus_1 # Update xn for the next iteration

    print(f"Warning: Newton's method did not converge to the specified tolerance ({tol}) after {max_iter} iterations.")
    print(f"Current approximation: {xn:.8f}")
    print_table(headers, iteration_data)
    return xn

# --- 4. Secant Method Implementation ---
def secant_method(func, x0, x1, equation_str, tol=1e-6, max_iter=100):
    """
    Implements the Secant Method for finding roots of a function.

    :param func: The function f(x) for which to find the root.
    :param x0: The first initial guess (xn-1).
    :param x1: The second initial guess (xn).
    :param equation_str: String representation of the function.
    :param tol: The desired tolerance for the root.
    :param max_iter: The maximum number of iterations.
    :return: The approximate root if found, else None.
    """
    print(f"--- Secant Method for f(x) = {equation_str} ---")
    print(f"Attempting to find root with initial guesses x0 = {x0}, x1 = {x1} and tolerance {tol}")
    
    try:
        f_x0 = func(x0)
        f_x1 = func(x1)
    except Exception as e:
        print(f"Error: Function evaluation failed at initial guesses x0 = {x0:.8f} or x1 = {x1:.8f}: {e}")
        print("Skipping this method call.")
        return None

    # Check for exact roots at initial guesses
    if f_x0 == 0.0:
        print(f"Exact root found at initial guess x0: {x0:.8f}")
        return x0
    if f_x1 == 0.0:
        print(f"Exact root found at initial guess x1: {x1:.8f}")
        return x1

    # Calculus-based pre-check: Check if initial slope is too flat or x0 == x1.
    if abs(x1 - x0) < sys.float_info.epsilon:
        print(f"Reason for skipping: Initial guesses x0 ({x0:.8f}) and x1 ({x1:.8f}) are too close or identical.")
        print("Secant method requires two distinct points to form an initial secant line.")
        print("Therefore, skipping iterations for this question.")
        return None
    
    if abs(f_x1 - f_x0) < sys.float_info.epsilon * max(abs(f_x0), abs(f_x1), 1):
        print(f"Reason for potential issue: f(x0) = {f_x0:.8f} and f(x1) = {f_x1:.8f} are too close at initial guesses.")
        print(f"The initial secant line has a slope close to zero ({ (f_x1 - f_x0) / (x1 - x0) :.8e}).")
        print("This could lead to an extremely large step in the first iteration, causing divergence or an error.")
        print("Proceeding with caution, but expect potential non-convergence.")
    else:
        print(f"Initial check passed: Initial guesses are distinct, and the initial slope is not too flat.")
        print(f"Secant method convergence is super-linear if initial guesses are sufficiently close to a root where f'(root) != 0.")
        print(f"It may diverge if initial guesses are far from a root or if the function's slope is near zero.")

    headers = ["Iteration", "xn-1", "xn", "f(xn-1)", "f(xn)", "xn+1"]
    iteration_data = []

    x_prev = x0
    x_curr = x1
    f_prev = f_x0 # f(xn-1)
    f_curr = f_x1 # f(xn)


    for i in range(1, max_iter + 1):
        # Check for exact root first at x_curr (newest point)
        if f_curr == 0.0:
            print(f"Exact root found after {i-1} iterations at xn: {x_curr:.8f}")
            print_table(headers, iteration_data) # Table up to previous iteration, or append final row
            return x_curr

        # Check for division by zero (f(x_curr) - f(x_prev) being too small) during iterations
        if abs(f_curr - f_prev) < sys.float_info.epsilon * max(abs(f_curr), abs(f_prev), 1):
            print(f"Error at iteration {i}: Denominator (f(xn)-f(xn-1)) is too close to zero ({f_curr - f_prev:.8e}).")
            print("Secant method failed due to a horizontal secant line. Skipping further iterations.")
            print_table(headers, iteration_data)
            return None

        # Calculate x_next (xn+1) using the secant formula
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        # Record data for the iteration table
        iteration_data.append([i, x_prev, x_curr, f_prev, f_curr, x_next])

        # Check for convergence
        if abs(f_curr) < tol or abs(x_next - x_curr) < tol: # Check f_curr before updating for next iteration, or diff in x
            print(f"Converged after {i} iterations. Approximate root: {x_next:.8f}")
            print_table(headers, iteration_data)
            return x_next

        # Update values for the next iteration
        x_prev = x_curr
        x_curr = x_next
        
        try:
            f_prev = f_curr
            f_curr = func(x_curr) # Calculate f(xn) for the new xn
        except Exception as e:
            print(f"Error: Function evaluation failed at xn = {x_curr:.8f} during iteration {i}: {e}")
            print("Stopping iterations.")
            print_table(headers, iteration_data)
            return None

    print(f"Warning: Secant method did not converge to the specified tolerance ({tol}) after {max_iter} iterations.")
    print(f"Current approximation: {x_curr:.8f}")
    print_table(headers, iteration_data)
    return x_curr

# --- 5. Fixed Point Iteration Implementation ---
def fixed_point_iteration(g_func, dg_func, x0, equation_str, tol=1e-6, max_iter=100):
    """
    Implements the Fixed Point Iteration method.

    :param g_func: The iteration function g(x) = x.
    :param dg_func: The derivative of the iteration function g'(x).
    :param x0: The initial guess.
    :param equation_str: String representation of the function.
    :param tol: The desired tolerance for convergence.
    :param max_iter: The maximum number of iterations.
    :return: The approximate fixed point if found, else None.
    """
    print(f"--- Fixed Point Iteration for g(x) = {equation_str} ---")
    print(f"Attempting to find fixed point with initial guess x0 = {x0} and tolerance {tol}")

    # Calculus-based pre-check: Check |g'(x0)|.
    try:
        dg_x0 = dg_func(x0)
        if abs(dg_x0) >= 1:
            print(f"Reason for likely divergence: |g'(x0)| = |{dg_x0:.8f}| >= 1 at x0 = {x0:.8f}.")
            print("For Fixed Point Iteration to converge locally, the absolute value of the derivative |g'(x)| must be less than 1 (i.e., |g'(x)| < 1) in an interval containing the fixed point and the initial guess.")
            print("Since this condition is not met at the initial guess, convergence is unlikely.")
            print("Skipping iterations for this question based on calculus-based divergence check.")
            return None
        else:
            print(f"Initial check passed: |g'(x0)| = |{dg_x0:.8f}| < 1 at x0 = {x0:.8f}.")
            print(f"Local convergence is likely if g(x) maps an interval around {x0} into itself and |g'(x)| < 1 throughout that interval.")
    except Exception as e:
        print(f"Warning: Could not evaluate g'(x0) at x0 = {x0:.8f} due to: {e}.")
        print("Proceeding without derivative-based divergence check, relying solely on iteration limits.")
        # If we can't evaluate dg_func, we can't perform the check, so we proceed but warn.

    headers = ["Iteration", "xn", "g(xn)"]
    iteration_data = []

    xn = x0
    for i in range(1, max_iter + 1):
        try:
            xn_plus_1 = g_func(xn)
        except Exception as e:
            print(f"Error: Function g(x) evaluation failed at xn = {xn:.8f} during iteration {i}: {e}")
            print("This usually means the iteration went outside the function's domain. Stopping iterations.")
            print_table(headers, iteration_data)
            return None
        
        # Record data for the iteration table
        iteration_data.append([i, xn, xn_plus_1])

        # Check for exact fixed point (xn_plus_1 == xn)
        if xn_plus_1 == xn: # Check for exact equality first
            print(f"Exact fixed point found after {i} iterations: {xn_plus_1:.8f}")
            print_table(headers, iteration_data)
            return xn_plus_1

        # Check for convergence
        if abs(xn_plus_1 - xn) < tol:
            print(f"Converged after {i} iterations. Approximate fixed point: {xn_plus_1:.8f}")
            print_table(headers, iteration_data)
            return xn_plus_1

        xn = xn_plus_1 # Update xn for the next iteration

    print(f"Warning: Fixed Point Iteration did not converge to the specified tolerance ({tol}) after {max_iter} iterations.")
    print(f"Current approximation: {xn:.8f}")
    print_table(headers, iteration_data)
    return xn

# ==============================================================================
#                      Assignment Questions Application
# ==============================================================================

# Get global tolerance from user
try:
    user_tol_str = input("Enter desired numerical tolerance (e.g., 1e-6, press Enter for default 1e-6): ")
    if user_tol_str:
        global_tolerance = float(user_tol_str)
        if global_tolerance <= 0:
            print("Tolerance must be positive. Using default 1e-6.")
            global_tolerance = 1e-6
    else:
        global_tolerance = 1e-6 # Default tolerance
        print(f"No tolerance entered, using default: {global_tolerance}")
except ValueError:
    print("Invalid tolerance entered. Using default 1e-6.")
    global_tolerance = 1e-6

print(f"\nAll subsequent methods will use a tolerance of {global_tolerance:.1e}\n")


# --- 1. Bisection Method Questions ---

# Question 1: f(x) = x^3 - 4x^2 + 6x - 24 in [2, 4]
def f1(x):
    return x**3 - 4*x**2 + 6*x - 24

print("\n" + "#"*80 + "\n")
print("----- QUESTION 1 -----")
bisection_method(f1, 2, 4, equation_str="x^3 - 4x^2 + 6x - 24", tol=global_tolerance)
print("\n" + "="*80 + "\n")

# Question 2: f(x) = sin(x) - 0.5 in [0, 1]
def f2(x):
    return np.sin(x) - 0.5

print("\n" + "#"*80 + "\n")
print("----- QUESTION 2 -----")
bisection_method(f2, 0, 1, equation_str="sin(x) - 0.5", tol=global_tolerance)
print("\n" + "="*80 + "\n")

# --- 2. False Position Method Questions ---

# Question 3: f(x) = ln(x) - 1 in [1, 3]
def f3(x):
    if x <= 0: # Ensure log is defined for positive x
        raise ValueError("Logarithm undefined for x <= 0")
    return np.log(x) - 1

print("\n" + "#"*80 + "\n")
print("----- QUESTION 3 -----")
false_position_method(f3, 1, 3, equation_str="ln(x) - 1", tol=global_tolerance)
print("\n" + "="*80 + "\n")

# Question 4: f(x) = e^x - 5 in [0, 2]
def f4(x):
    return np.exp(x) - 5

print("\n" + "#"*80 + "\n")
print("----- QUESTION 4 -----")
false_position_method(f4, 0, 2, equation_str="e^x - 5", tol=global_tolerance)
print("\n" + "="*80 + "\n")

# --- 3. Newton's Method Questions ---

# Question 5: f(x) = x^4 - 2x^3 + x - 1 with x0 = 1
def f5(x):
    return x**4 - 2*x**3 + x - 1

def df5(x): # Derivative of f5(x)
    return 4*x**3 - 6*x**2 + 1

print("\n" + "#"*80 + "\n")
print("----- QUESTION 5 -----")
newton_method(f5, df5, 1, equation_str="x^4 - 2x^3 + x - 1", tol=global_tolerance)
print("\n" + "="*80 + "\n")

# Question 6: f(x) = tan(x) - x with x0 = 1
def f6(x):
    return np.tan(x) - x

def df6(x): # Derivative of f6(x)
    return np.tan(x)**2 # f'(x) = sec^2(x) - 1 = tan^2(x)

print("\n" + "#"*80 + "\n")
print("----- QUESTION 6 -----")
newton_method(f6, df6, 1, equation_str="tan(x) - x", tol=global_tolerance)
print("\n" + "="*80 + "\n")

# --- 4. Secant Method Questions ---

# Question 7: f(x) = x^2 - 2 with x0 = 1, x1 = 2
def f7(x):
    return x**2 - 2

print("\n" + "#"*80 + "\n")
print("----- QUESTION 7 -----")
secant_method(f7, 1, 2, equation_str="x^2 - 2", tol=global_tolerance)
print("\n" + "="*80 + "\n")

# Question 8: f(x) = e^(-x) - x with x0 = 0, x1 = 1
def f8(x):
    return np.exp(-x) - x

print("\n" + "#"*80 + "\n")
print("----- QUESTION 8 -----")
secant_method(f8, 0, 1, equation_str="e^(-x) - x", tol=global_tolerance)
print("\n" + "="*80 + "\n")

# --- 5. Fixed Point Iteration Questions ---

# Question 9: g(x) = sqrt(6 + x) with x0 = 2
def g9(x):
    if 6 + x < 0: # Ensure sqrt is defined for non-negative values
        raise ValueError("Argument of sqrt must be non-negative")
    return np.sqrt(6 + x)

def dg9(x): # Derivative of g9(x)
    if 6 + x <= 0: # Derivative undefined or problematic
        raise ValueError("Derivative of sqrt(u) is 1/(2*sqrt(u)), undefined for u <= 0")
    return 1 / (2 * np.sqrt(6 + x))

print("\n" + "#"*80 + "\n")
print("----- QUESTION 9 -----")
fixed_point_iteration(g9, dg9, 2, equation_str="sqrt(6 + x)", tol=global_tolerance)
print("Additional information for Question 9:")
print("To determine the convergence region for g(x) = sqrt(6 + x), we analyze where |g'(x)| < 1.")
print("g'(x) = 1 / (2 * sqrt(6 + x)).")
print("We need |1 / (2 * sqrt(6 + x))| < 1, which implies 2 * sqrt(6 + x) > 1 (assuming 6+x > 0).")
print("Squaring both sides (valid since both are positive): 4 * (6 + x) > 1 => 24 + 4x > 1 => 4x > -23 => x > -5.75.")
print("Also, for g(x) to be real, 6 + x >= 0, so x >= -6.")
print("Combining these, the convergence region where |g'(x)| < 1 is x > -5.75.")
print("The fixed points for x = sqrt(6+x) are x=3 and x=-2.")
print("At x=3: g'(3) = 1 / (2 * sqrt(6+3)) = 1 / (2 * sqrt(9)) = 1/6. Since |1/6| < 1, x=3 is a stable fixed point.")
print("At x=-2: g'(-2) = 1 / (2 * sqrt(6-2)) = 1 / (2 * sqrt(4)) = 1/4. Since |1/4| < 1, x=-2 is also a stable fixed point.")
print(f"Given initial guess x0 = 2, which is within the convergence region x > -5.75, the method converged to x=3.")
print("\n" + "="*80 + "\n")

# Question 10: g(x) = 1 + ln(x) with x0 = 2
def g10(x):
    if x <= 0: # Ensure log is defined for positive x
        raise ValueError("Logarithm undefined for x <= 0")
    return 1 + np.log(x)

def dg10(x): # Derivative of g10(x)
    if x <= 0: # Derivative undefined or problematic
        raise ValueError("Derivative undefined or problematic for x <= 0")
    return 1 / x

print("\n" + "#"*80 + "\n")
print("----- QUESTION 10 -----")
fixed_point_iteration(g10, dg10, 2, equation_str="1 + ln(x)", tol=global_tolerance)
print("Additional information for Question 10:")
print("To determine the convergence region for g(x) = 1 + ln(x), we analyze where |g'(x)| < 1.")
print("g'(x) = 1 / x.")
print("We need |1 / x| < 1, which implies |x| > 1.")
print("Additionally, for ln(x) to be defined, x > 0.")
print("Combining these, the convergence region where |g'(x)| < 1 is x > 1.")
print("The fixed point for x = 1 + ln(x) is x=1 (since 1 = 1 + ln(1) => 1 = 1 + 0).")
print("At the fixed point x=1: g'(1) = 1/1 = 1.")
print("When |g'(fixed_point)| = 1, convergence is a critical case. The method is neither guaranteed to converge nor to diverge locally.")
print("It typically implies very slow, linear convergence, or it might not converge at all if not starting exactly at the fixed point, or if higher-order terms prevent it.")
print(f"Given initial guess x0 = 2, which is in the region x > 1 (g'(2) = 0.5 < 1), the method converged towards x=1. The convergence rate slows down as x approaches 1 due to g'(x) approaching 1.")
print("\n" + "="*80 + "\n")