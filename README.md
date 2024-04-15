
# Probabilistic graph implementation

* Main ideia summary of a logistic graph

## Main equation:
$$
prob_of_vertex_{i,j} =
\frac{ 1 }
{ 1 + \exp^( coef_0 |i| + coef_1 |j| + intercept)  }
$$

1. First of all initialize n (number of vertices), c (1e-3 initial prob), beta( low value )
2. Run with logistic regression above to check if 2 vertex are going to connect
3. Stop with convergence. Here the convergence criterion is not well defined,
at first.
4. Estimate c_hat and beta_hat and check if the estimation is any good
5. After that repeat the above with a more parameters. The parameters are any
vertex that is conenected to i any vertex connected (including j), any vertex connected to j
(including k) and so on. Repeat with p <= 4 whrere p i the order of the algo

### Objective:
To model the probability of a connection between two vertices in a graph using a logistic function.

### Equation:
$$
\text{prob\_of\_vertex}_{i,j} = \frac{c}{1 + \beta \exp(|i| + |j|)}
$$

Where:
- \( \text{prob\_of\_vertex}_{i,j} \) is the probability of a connection between vertices \( i \) and \( j \).
- \( c \) is the initial probability.
- \( \beta \) is a parameter that modulates the exponential term.

### Steps:
1. **Initialization**:
   - Set \( n \): Number of vertices.
   - Set \( c \): Initial probability (e.g., \( 1 \times 10^{-3} \)).
   - Set \( \beta \): A low value to start with.
  
2. **Logistic Regression**:
   - Use the given logistic function to compute the probability of a connection between vertices \( i \) and \( j \).

3. **Convergence**:
   - Monitor the model for convergence. The exact criterion for convergence isn't predefined.

4. **Parameter Estimation**:
   - Estimate \( c_{\text{hat}} \) and \( \beta_{\text{hat}} \).
   - Assess the goodness of fit of the estimations.

5. **Iterative Refinement**:
   - Incorporate more parameters based on the connectivity of the vertices.
   - For instance, for a vertex \( i \), consider all vertices connected to \( i \), and for vertex \( j \), consider all vertices connected to \( j \) and so on.
   - Repeat the logistic regression with increasing order \( p \) (up to \( p \leq 4 \)).

### Note:
The methodology seems to be an iterative approach to improve the modeling of vertex connectivity. The inclusion of more parameters based on connected vertices in step 5 suggests that the model may try to capture higher-order relationships or dependencies between vertices in the graph.