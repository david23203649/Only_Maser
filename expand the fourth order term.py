import sympy as sp

# Define the symbols
alpha = sp.symbols('alpha')
q, q_dagger = sp.symbols('q q_dagger', commutative=False)
s, s_dagger = sp.symbols('s s_dagger', commutative=False)
m, m_dagger = sp.symbols('m m_dagger', commutative=False)
lambda_qs, lambda_qm,  eta, eta_star = sp.symbols('lambda_qs lambda_qm eta eta_star')
omega_q, omega_s, omega_p, omega_m = sp.symbols('omega_q omega_s omega_p omega_m', real=True)

# Time symbol
t = sp.symbols('t')

# Define the components of the expression
term1 = q * sp.exp(-sp.I * omega_q * t)
term2 = -lambda_qs * s * sp.exp(-sp.I * omega_s * t)
term3 = lambda_qm * m * sp.exp(-sp.I * omega_q * t)
term4 = q_dagger * sp.exp(sp.I * omega_q * t)
term5 = - lambda_qs * sp.exp(sp.I * omega_s * t)
term6 =  lambda_qm * m_dagger * sp.exp(sp.I * omega_q * t)
term7 = lambda_qs * eta * sp.exp(-sp.I * omega_p * t)
term8 =  lambda_qs * eta_star * sp.exp(sp.I * omega_p * t)
# Now construct the expression
expr =  (term1 + term2 )**2



# Expand the expression
expanded_expr = sp.expand(expr)


# Print the expanded result
sp.pprint(expanded_expr)
simplified_expr = sp.simplify(expanded_expr)
print("----after simplified-----")
sp.pprint(simplified_expr)