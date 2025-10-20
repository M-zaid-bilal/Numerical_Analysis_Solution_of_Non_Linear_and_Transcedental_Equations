# Numerical Analysis - Assignment 1
## Solution of Non-linear and Transcendental Equations

**By:** Muhammad Zaid Bilal  
**ERP:** 30506  
**Tolerance:** 1.0e-06

---

## Question 1: Bisection Method
**Function:** f(x) = x³ - 4x² + 6x - 24  
**Interval:** [2, 4]

### Result
**Exact root found at initial upper bound:** 4.00000000

---

## Question 2: Bisection Method
**Function:** f(x) = sin(x) - 0.5  
**Interval:** [0, 1]

### Result
- **Converged after 17 iterations**
- **Approximate root:** 0.52359772
- Initial check: f(a) = -0.500000 and f(b) = 0.341471 have opposite signs

### Iteration Table

| Iteration | a          | b          | c          | f(a)        | f(b)       | f(c)        |
|-----------|------------|------------|------------|-------------|------------|-------------|
| 1         | 0          | 1          | 0.50000000 | -0.50000000 | 0.34147098 | -0.02057446 |
| 2         | 0.50000000 | 1          | 0.75000000 | -0.02057446 | 0.34147098 | 0.18163876  |
| 3         | 0.50000000 | 0.75000000 | 0.62500000 | -0.02057446 | 0.18163876 | 0.08509727  |
| 4         | 0.50000000 | 0.62500000 | 0.56250000 | -0.02057446 | 0.08509727 | 0.03330267  |
| 5         | 0.50000000 | 0.56250000 | 0.53125000 | -0.02057446 | 0.03330267 | 0.00661145  |
| 6         | 0.50000000 | 0.53125000 | 0.51562500 | -0.02057446 | 0.00661145 | -0.00692131 |
| 7         | 0.51562500 | 0.53125000 | 0.52343750 | -0.00692131 | 0.00661145 | -0.00013968 |
| 8         | 0.52343750 | 0.53125000 | 0.52734375 | -0.00013968 | 0.00661145 | 0.00323973  |
| 9         | 0.52343750 | 0.52734375 | 0.52539062 | -0.00013968 | 0.00323973 | 0.00155098  |
| 10        | 0.52343750 | 0.52539062 | 0.52441406 | -0.00013968 | 0.00155098 | 0.00070589  |
| 11        | 0.52343750 | 0.52441406 | 0.52392578 | -0.00013968 | 0.00070589 | 0.00028317  |
| 12        | 0.52343750 | 0.52392578 | 0.52368164 | -0.00013968 | 0.00028317 | 0.00007176  |
| 13        | 0.52343750 | 0.52368164 | 0.52355957 | -0.00013968 | 0.00007176 | -0.00003395 |
| 14        | 0.52355957 | 0.52368164 | 0.52362061 | -0.00003395 | 0.00007176 | 0.00001891  |
| 15        | 0.52355957 | 0.52362061 | 0.52359009 | -0.00003395 | 0.00001891 | -0.00000752 |
| 16        | 0.52359009 | 0.52362061 | 0.52360535 | -0.00000752 | 0.00001891 | 0.00000569  |
| 17        | 0.52359009 | 0.52360535 | 0.52359772 | -0.00000752 | 0.00000569 | -0.00000092 |

---

## Question 3: False Position Method
**Function:** f(x) = ln(x) - 1  
**Interval:** [1, 3]

### Result
- **Converged after 12 iterations**
- **Approximate root:** 2.71828352 (≈ e)
- Convergence is generally guaranteed, potentially faster than bisection

### Iteration Table

| Iteration | a_n | b_n        | f(a_n)      | f(b_n)     | c_n        | f(c_n)     |
|-----------|-----|------------|-------------|------------|------------|------------|
| 1         | 1   | 3          | -1.00000000 | 0.09861229 | 2.82047845 | 0.03690653 |
| 2         | 1   | 2.82047845 | -1.00000000 | 0.03690653 | 2.75568230 | 0.01366507 |
| 3         | 1   | 2.75568230 | -1.00000000 | 0.01366507 | 2.73201420 | 0.00503914 |
| 4         | 1   | 2.73201420 | -1.00000000 | 0.00503914 | 2.72333010 | 0.00185543 |
| 5         | 1   | 2.72333010 | -1.00000000 | 0.00185543 | 2.72013850 | 0.00068280 |
| 6         | 1   | 2.72013850 | -1.00000000 | 0.00068280 | 2.71896479 | 0.00025122 |
| 7         | 1   | 2.71896479 | -1.00000000 | 0.00025122 | 2.71853307 | 0.00009242 |
| 8         | 1   | 2.71853307 | -1.00000000 | 0.00009242 | 2.71837425 | 0.00003400 |
| 9         | 1   | 2.71837425 | -1.00000000 | 0.00003400 | 2.71831583 | 0.00001251 |
| 10        | 1   | 2.71831583 | -1.00000000 | 0.00001251 | 2.71829434 | 0.00000460 |
| 11        | 1   | 2.71829434 | -1.00000000 | 0.00000460 | 2.71828643 | 0.00000169 |
| 12        | 1   | 2.71828643 | -1.00000000 | 0.00000169 | 2.71828352 | 0.00000062 |

---

## Question 4: False Position Method
**Function:** f(x) = e^x - 5  
**Interval:** [0, 2]

### Result
- **Converged after 10 iterations**
- **Approximate root:** 1.60943783 (≈ ln(5))

### Iteration Table

| Iteration | a_n        | b_n | f(a_n)      | f(b_n)     | c_n        | f(c_n)      |
|-----------|------------|-----|-------------|------------|------------|-------------|
| 1         | 0          | 2   | -4.00000000 | 2.38905610 | 1.25214114 | -1.50217572 |
| 2         | 1.25214114 | 2   | -1.50217572 | 2.38905610 | 1.54084546 | -0.33146435 |
| 3         | 1.54084546 | 2   | -0.33146435 | 2.38905610 | 1.59678819 | -0.06285023 |
| 4         | 1.59678819 | 2   | -0.06285023 | 2.38905610 | 1.60712381 | -0.01155715 |
| 5         | 1.60712381 | 2   | -0.01155715 | 2.38905610 | 1.60901521 | -0.00211306 |
| 6         | 1.60901521 | 2   | -0.00211306 | 2.38905610 | 1.60936072 | -0.00038594 |
| 7         | 1.60936072 | 2   | -0.00038594 | 2.38905610 | 1.60942382 | -0.00007048 |
| 8         | 1.60942382 | 2   | -0.00007048 | 2.38905610 | 1.60943534 | -0.00001287 |
| 9         | 1.60943534 | 2   | -0.00001287 | 2.38905610 | 1.60943744 | -0.00000235 |
| 10        | 1.60943744 | 2   | -0.00000235 | 2.38905610 | 1.60943783 | -0.00000043 |

---

## Question 5: Newton's Method
**Function:** f(x) = x⁴ - 2x³ + x - 1  
**Initial Guess:** x₀ = 1

### Result
⚠️ **WARNING:** Did not converge after 100 iterations  
**Current approximation:** 1.00000000

### Analysis
The method oscillates between x = 0 and x = 1, indicating:
- The initial guess leads to a cycle
- f(0) = -1 and f(1) = -1 (both negative)
- The method requires a different initial guess for convergence

### Sample Iterations

| Iteration | xₙ         | f(xₙ)       | f'(xₙ)      | xₙ₊₁       |
|-----------|------------|-------------|-------------|------------|
| 1         | 1          | -1          | -1          | 0.00000000 |
| 2         | 0.00000000 | -1.00000000 | 1.00000000  | 1.00000000 |
| 3         | 1.00000000 | -1.00000000 | -1.00000000 | 0.00000000 |
| ...       | ...        | ...         | ...         | ...        |

*(Pattern continues indefinitely)*

---

## Question 6: Newton's Method
**Function:** f(x) = tan(x) - x  
**Initial Guess:** x₀ = 1

### Result
- **Converged after 13 iterations**
- **Approximate root:** 0.00698419

### Iteration Table

| Iteration | xₙ         | f(xₙ)      | f'(xₙ)     | xₙ₊₁       |
|-----------|------------|------------|------------|------------|
| 1         | 1          | 0.55740772 | 2.42551882 | 0.77019031 |
| 2         | 0.77019031 | 0.19984734 | 0.94097304 | 0.55780661 |
| 3         | 0.55780661 | 0.06609158 | 0.38924895 | 0.38801404 |
| 4         | 0.38801404 | 0.02072125 | 0.16706453 | 0.26398265 |
| 5         | 0.26398265 | 0.00630793 | 0.07305700 | 0.17764008 |
| 6         | 0.17764008 | 0.00189243 | 0.03223192 | 0.11892725 |
| 7         | 0.11892725 | 0.00056388 | 0.01427813 | 0.07943465 |
| 8         | 0.07943465 | 0.00016750 | 0.00633650 | 0.05300103 |
| 9         | 0.05300103 | 0.00004968 | 0.00281438 | 0.03534726 |
| 10        | 0.03534726 | 0.00001473 | 0.00125047 | 0.02356877 |
| 11        | 0.02356877 | 0.00000437 | 0.00055569 | 0.01571367 |
| 12        | 0.01571367 | 0.00000129 | 0.00024696 | 0.01047613 |
| 13        | 0.01047613 | 0.00000038 | 0.00010976 | 0.00698419 |

---

## Question 7: Secant Method
**Function:** f(x) = x² - 2  
**Initial Guesses:** x₀ = 1, x₁ = 2

### Result
- **Converged after 6 iterations**
- **Approximate root:** 1.41421356 (≈ √2)

### Iteration Table

| Iteration | xₙ₋₁       | xₙ         | f(xₙ₋₁)     | f(xₙ)       | xₙ₊₁       |
|-----------|------------|------------|-------------|-------------|------------|
| 1         | 1          | 2          | -1          | 2           | 1.33333333 |
| 2         | 2          | 1.33333333 | 2           | -0.22222222 | 1.40000000 |
| 3         | 1.33333333 | 1.40000000 | -0.22222222 | -0.04000000 | 1.41463415 |
| 4         | 1.40000000 | 1.41463415 | -0.04000000 | 0.00118977  | 1.41421144 |
| 5         | 1.41463415 | 1.41421144 | 0.00118977  | -0.00000601 | 1.41421356 |
| 6         | 1.41421144 | 1.41421356 | -0.00000601 | -0.00000000 | 1.41421356 |

---

## Question 8: Secant Method
**Function:** f(x) = e⁻ˣ - x  
**Initial Guesses:** x₀ = 0, x₁ = 1

### Result
- **Converged after 5 iterations**
- **Approximate root:** 0.56714329

### Iteration Table

| Iteration | xₙ₋₁       | xₙ         | f(xₙ₋₁)     | f(xₙ)       | xₙ₊₁       |
|-----------|------------|------------|-------------|-------------|------------|
| 1         | 0          | 1          | 1.00000000  | -0.63212056 | 0.61269984 |
| 2         | 1          | 0.61269984 | -0.63212056 | -0.07081395 | 0.56383839 |
| 3         | 0.61269984 | 0.56383839 | -0.07081395 | 0.00518235  | 0.56717036 |
| 4         | 0.56383839 | 0.56717036 | 0.00518235  | -0.00004242 | 0.56714331 |
| 5         | 0.56717036 | 0.56714331 | -0.00004242 | -0.00000003 | 0.56714329 |

---

## Question 9: Fixed Point Iteration
**Function:** g(x) = √(6 + x)  
**Initial Guess:** x₀ = 2

### Result
- **Converged after 9 iterations**
- **Approximate fixed point:** 2.99999990 (≈ 3)
- |g'(x₀)| = 0.17677670 < 1, ensuring local convergence

### Iteration Table

| Iteration | xₙ         | g(xₙ)      |
|-----------|------------|------------|
| 1         | 2          | 2.82842712 |
| 2         | 2.82842712 | 2.97126692 |
| 3         | 2.97126692 | 2.99520733 |
| 4         | 2.99520733 | 2.99920111 |
| 5         | 2.99920111 | 2.99986685 |
| 6         | 2.99986685 | 2.99997781 |
| 7         | 2.99997781 | 2.99999630 |
| 8         | 2.99999630 | 2.99999938 |
| 9         | 2.99999938 | 2.99999990 |

### Convergence Analysis
**Derivative:** g'(x) = 1 / (2√(6 + x))

**Convergence Condition:** |g'(x)| < 1
- Requires: 2√(6 + x) > 1
- Simplifies to: x > -5.75
- Combined with domain requirement (x ≥ -6): **x > -5.75**

**Fixed Points:**
- x = 3: g'(3) = 1/6, stable (|g'(3)| < 1)
- x = -2: g'(-2) = 1/4, stable (|g'(-2)| < 1)

With x₀ = 2, the method converged to x = 3.

---

## Question 10: Fixed Point Iteration
**Function:** g(x) = 1 + ln(x)  
**Initial Guess:** x₀ = 2

### Result
⚠️ **WARNING:** Did not converge after 100 iterations  
**Current approximation:** 1.01985723  
- |g'(x₀)| = 0.50000000 < 1 at x₀ = 2

### Sample Iterations

| Iteration | xₙ         | g(xₙ)      |
|-----------|------------|------------|
| 1         | 2          | 1.69314718 |
| 2         | 1.69314718 | 1.52658903 |
| 3         | 1.52658903 | 1.42303586 |
| 10        | 1.19104169 | 1.17482829 |
| 20        | 1.09875815 | 1.09418059 |
| 30        | 1.06640093 | 1.06428936 |
| 50        | 1.04003831 | 1.03925755 |
| 75        | 1.02673158 | 1.02638054 |
| 100       | 1.01985723 | 1.01966270 |

### Convergence Analysis
**Derivative:** g'(x) = 1/x

**Convergence Condition:** |g'(x)| < 1
- Requires: |x| > 1
- Combined with domain requirement (x > 0): **x > 1**

**Fixed Point:** x = 1
- At x = 1: g'(1) = 1

**Critical Case:** When |g'(fixed_point)| = 1
- Convergence is neither guaranteed nor ruled out
- Typically results in very slow linear convergence
- Method approaches the fixed point but extremely slowly

With x₀ = 2 (where g'(2) = 0.5 < 1), the method converges toward x = 1, but convergence slows significantly as x approaches 1 due to g'(x) → 1.

---

## Summary of Results

| Question | Method              | Function                        | Result                | Iterations | Status      |
|----------|---------------------|----------------------------------|----------------------|------------|-------------|
| 1        | Bisection           | x³ - 4x² + 6x - 24              | 4.00000000           | 0          | ✓ Exact     |
| 2        | Bisection           | sin(x) - 0.5                    | 0.52359772           | 17         | ✓ Converged |
| 3        | False Position      | ln(x) - 1                       | 2.71828352           | 12         | ✓ Converged |
| 4        | False Position      | eˣ - 5                          | 1.60943783           | 10         | ✓ Converged |
| 5        | Newton's            | x⁴ - 2x³ + x - 1                | 1.00000000           | 100        | ✗ Failed    |
| 6        | Newton's            | tan(x) - x                      | 0.00698419           | 13         | ✓ Converged |
| 7        | Secant              | x² - 2                          | 1.41421356           | 6          | ✓ Converged |
| 8        | Secant              | e⁻ˣ - x                         | 0.56714329           | 5          | ✓ Converged |
| 9        | Fixed Point         | g(x) = √(6 + x)                 | 2.99999990           | 9          | ✓ Converged |
| 10       | Fixed Point         | g(x) = 1 + ln(x)                | 1.01985723           | 100        | ⚠ Slow      |

---

## Conclusions

1. **Bisection Method**: Reliable but requires more iterations; guaranteed convergence when initial interval brackets a root.

2. **False Position Method**: Generally faster than bisection; performed well for both logarithmic and exponential functions.

3. **Newton's Method**: Fastest convergence when it works, but sensitive to initial guess. Question 5 demonstrates divergence due to poor initial guess leading to oscillation.

4. **Secant Method**: Super-linear convergence without requiring derivatives; performed excellently for both test cases.

5. **Fixed Point Iteration**: Convergence depends critically on |g'(x)| < 1. Question 9 converged rapidly, while Question 10 showed very slow convergence due to g'(x) → 1 at the fixed point.
