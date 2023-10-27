
* Main ideia summary of a logistic graph

Main equation:

$$
prob of vertex i,j =
\frac{ c }
{ 1 + \beta \exp^( |i| + |j| )  }
$$

    1. First of all initialize n (number of vertices), c (1e-3 initial prob), beta( low value )
    2. Run with logistic regression above to check if 2 vertex are going to connect
    3. Stop with convergence. Here the convergence criterion is not well defined,
    at first.
    4. Estimate c_hat and beta_hat and check if the estimation is any good
    5. After that repeat the above with a more parameters. The parameters are any
    vertex that is conenected to i any vertex connected (including j), any vertex connected to j
    (including k) and so on. Repeat with p <= 4 whrere p i the order of the algo
