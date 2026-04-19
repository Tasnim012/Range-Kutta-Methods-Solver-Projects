import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import sympify, symbols, lambdify

st.set_page_config(page_title="RK4 Step-by-Step Solver", layout="wide")

st.title("Runge-Kutta 4th Order Step-by-Step Solver", anchor=False)

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")

    eq_str = st.text_input("Enter f(x, y):", placeholder="e.g. x + y**2")
    x0 = st.number_input("Initial x (x0):", placeholder="Enter value of x (x0)")
    y0 = st.number_input("Initial y (y0):", placeholder="Enter value of y (y0)")
    h = st.number_input("Step size (h):", placeholder="Enter value of (h)")
    steps = st.number_input("Number of steps:", value=2, min_value=1)

# Logic with Validation
if not eq_str:
    st.warning("Please enter an equation (e.g., x + y**2) in the sidebar to start.")
else:
    try:
        x_sym, y_sym = symbols('x y')
        expr = sympify(eq_str)
        f_func = lambdify((x_sym, y_sym), expr)

        #Make "Given Data" section
        st.subheader("SOLUTION")
        st.markdown(f"**Given data:** $f(x, y) = {eq_str}$, $h = {h}$")
        
        cols = st.columns(int(steps) + 1)
        with cols[0]:
            st.latex(f"x_0 = {x0}")
            st.latex(f"y_0 = {y0}")
        
        for i in range(1, int(steps) + 1):
            with cols[i]:
                target_x = round(x0 + i*h, 2)
                st.latex(f"x_{i} = {target_x}")
                st.latex(f"y_{i} = ?")
        
        st.markdown("---")

        curr_x, curr_y = x0, y0
        x_plot = [x0]
        y_plot = [y0]

        # Iteration loop
        for i in range(int(steps)):
            target_x = round(curr_x + h, 2)
            st.subheader(f"Putting $n = {i}$ in the Runge-Kutta method of 4th order:")
            st.latex(f"y_{{{i+1}}} = y_{i} + \\frac{{1}}{{6}}[k_1 + 2k_2 + 2k_3 + k_4] \quad ...... (i)")
            st.write("**Where,**")
            
            k1 = h * f_func(curr_x, curr_y)
            k2 = h * f_func(curr_x + h/2, curr_y + k1/2)
            k3 = h * f_func(curr_x + h/2, curr_y + k2/2)
            k4 = h * f_func(curr_x + h, curr_y + k3)
            
            y_next = curr_y + (k1 + 2*k2 + 2*k3 + k4) / 6
            
            # Formatting for display
            st.latex(f"k_1 = h f(x_{i}, y_{i}) = {round(k1, 5)}")
            st.latex(f"k_2 = h f(x_{i} + \\frac{{h}}{{2}}, y_{i} + \\frac{{k_1}}{{2}}) = {round(k2, 5)}")
            st.latex(f"k_3 = h f(x_{i} + \\frac{{h}}{{2}}, y_{i} + \\frac{{k_2}}{{2}}) = {round(k3, 5)}")
            st.latex(f"k_4 = h f(x_{i} + h, y_{i} + k_3) = {round(k4, 5)}")
            
            st.write("**Putting all values in the equation (i):**")
            st.latex(f"y_{{{i+1}}} = {round(curr_y, 4)} + \\frac{{1}}{{6}}[{round(k1, 4)} + 2({round(k2, 4)}) + 2({round(k3, 4)}) + {round(k4, 4)}] = {round(y_next, 5)}")
            
            st.success(f"Result: $y({target_x}) = {round(y_next, 5)}$")
            
            curr_x = target_x
            curr_y = y_next
            x_plot.append(curr_x)
            y_plot.append(curr_y)
            st.markdown("---")

        # Graph
        st.subheader("Solution Curve")
        fig, ax = plt.subplots()
        ax.plot(x_plot, y_plot, marker='o', color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"**Syntax Error!** Please check your equation. Use '*' for multiplication and '**' for power.")
        st.info("Example: 'x + y**2' instead of 'x + y2'")