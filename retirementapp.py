import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Dynamic Financial Planner")

# Load and display the image
banner_image = Image.open("retirement_banner.png")
st.image(banner_image, use_container_width=True)

st.title("Dynamic Financial Planner")
st.subheader("This application may be used to plan and prepare the retirement of a hypothetical client.")

# Section 1: Monthly Income
st.header("**Monthly Income**")
st.subheader("First, what is their expected annual income?")

colIncome1, colIncome2 = st.columns(2)

with colIncome1:
    salary = st.number_input("Enter your annual salary($): ", min_value=0.0, format='%f')
    tax_rate = st.number_input("Enter your tax rate(%): ", min_value=0.0, format='%f')

with colIncome2:
    promotion_rate = st.number_input("Enter promotion percentage increase(%):", min_value=0.0, value=5.0, step=0.5, format='%f')
    promotion_frequency = st.number_input("Enter promotion frequency (years):", min_value=1, value=3, step=1, format='%d')

# Default values for variables to avoid undefined errors
career_years = 0
projected_salaries = []
fig_salary_growth = go.Figure()

# Compute monthly income considering promotions
if salary > 0:
    tax_rate = tax_rate / 100.0
    salary_after_taxes = salary * (1 - tax_rate)
    monthly_takehome_salary = round(salary_after_taxes / 12.0, 2)

    # Apply promotions over career span
    career_years = st.slider("Number of years until retirement:", min_value=1, max_value=50, value=30, step=1)
    projected_salaries = [salary_after_taxes]
    
    for year in range(1, career_years + 1):
        if year % promotion_frequency == 0:
            projected_salaries.append(projected_salaries[-1] * (1 + promotion_rate / 100))
        else:
            projected_salaries.append(projected_salaries[-1])

    # Calculate average monthly take-home salary over career
    avg_monthly_takehome = round(np.mean(projected_salaries) / 12, 2)
else:
    salary_after_taxes = 0
    monthly_takehome_salary = 0
    avg_monthly_takehome = 0

st.subheader(f"Current Monthly Take-Home Salary: ${monthly_takehome_salary}")
st.subheader(f"Average Monthly Take-Home Salary Over Career (with promotions): ${avg_monthly_takehome}")

# Visualize projected salaries over career
st.subheader("Projected Salary Growth Over Career")
fig_salary_growth = go.Figure()
fig_salary_growth.add_trace(go.Scatter(
    x=list(range(career_years + 1)),
    y=projected_salaries,
    mode='lines+markers',
    name='Projected Annual Salary (After Taxes)'
))
fig_salary_growth.update_layout(
    title="Projected Annual Salary Growth with Promotions",
    xaxis_title="Years",
    yaxis_title="Annual Salary ($)"
)
st.plotly_chart(fig_salary_growth, use_container_width=True)

# Section 2: Monthly Expenses
st.header("**Monthly Expenses**")
st.subheader("Next, we can plot out planned expenses.")
colExpense1, colExpense2 = st.columns(2)

with colExpense1:
    monthly_rent = st.number_input("Enter your monthly rent($): ", min_value=0.0, format='%f')
    food_budget = st.number_input("Enter your monthly food budget($): ", min_value=0.0, format='%f')

with colExpense2:
    transport = st.number_input("Enter your monthly transport cost($): ", min_value=0.0, format='%f')
    entertainment = st.number_input("Enter your monthly entertainment budget($): ", min_value=0.0, format='%f')

monthly_expenses = monthly_rent + food_budget + transport + entertainment
monthly_savings = round(monthly_takehome_salary - monthly_expenses, 2) if monthly_takehome_salary > 0 else 0

st.subheader(f"Total Monthly Expenses: ${monthly_expenses}")

# Section 3: Retirement Planning
st.header("**Retirement Planning**")
st.subheader("Based on income and expenses, what should the expected retirement contributions be?")
colRetire1, colRetire2 = st.columns(2)

with colRetire1:
    current_investments = st.number_input("Current Investments($): ", min_value=0.0, format='%f')
    default_contribution = 2000.0 if monthly_savings == 0 else min(2000.0, monthly_savings)
    monthly_contribution = st.slider(
        "Adjust Monthly Contribution to Investments($):",
        min_value=0.0,
        max_value=float(monthly_savings) if monthly_savings > 0 else 2000.0,
        value=default_contribution,
        step=50.0
    )

with colRetire2:
    future_savings_goal = st.number_input("Future Savings Goal($): ", min_value=0.0, format='%f')
    annual_return = st.slider("Expected Annual Return on Investments(%):", min_value=0.0, max_value=15.0, value=10.0, step=0.5) / 100

future_investments = current_investments * (1 + annual_return)**career_years + \
                     monthly_contribution * (((1 + annual_return/12)**(career_years * 12) - 1) / (annual_return / 12))


goal_met = future_investments >= future_savings_goal
indicator_color = "green" if goal_met else "red"
st.markdown(f"**Projected Investments at Retirement:** <span style='color:{indicator_color}'>${round(future_investments, 2)}</span>", unsafe_allow_html=True)

# Section 4: Debt Repayment
st.header("**Debt Repayment**")
st.subheader("Are there any lingering debts?")
colDebt1, colDebt2 = st.columns(2)

with colDebt1:
    debt_balance = st.number_input("Enter your current debt balance($): ", min_value=0.0, format='%f')
    debt_interest_rate = st.slider("Debt Interest Rate(%):", min_value=0.0, max_value=20.0, value=5.0, step=0.5) / 100

with colDebt2:
    monthly_debt_payment = st.slider("Monthly Debt Payment($):", min_value=0.0, max_value=2000.0, value=300.0, step=50.0)

months_to_payoff = 0
remaining_debt = debt_balance
while remaining_debt > 0:
    interest = remaining_debt * (debt_interest_rate / 12)
    remaining_debt += interest - monthly_debt_payment
    months_to_payoff += 1

st.subheader(f"Estimated Months to Pay Off Debt: {months_to_payoff}")

# Section 5: Visualization
st.header("**Visualization**")

# Pie Chart for Expense Breakdown
st.subheader("Expense Breakdown")
expense_labels = ['Rent', 'Food', 'Transport', 'Entertainment', 'Debt Payment']
expense_values = [monthly_rent, food_budget, transport, entertainment, monthly_debt_payment]
fig_expense_pie = px.pie(values=expense_values, names=expense_labels, title="Monthly Expenditures")
st.plotly_chart(fig_expense_pie, use_container_width=True)

# Section 6: Risk-Based Investment Scenarios
st.header("**Retirement Scenarios**")
st.subheader("Based on the above assumptions, we can begin to plot the expected wealth accumulation over a career.")

tab1, tab2 = st.tabs(["Risk-Oriented Scenarios", "S&P 500-Based Scenarios"])

with tab1:
    st.markdown("""
    ### Risk-Oriented Scenarios
    - **Risk-Loving:** Assumes a higher return rate (12%) but with increased volatility.
    - **Risk-Neutral:** Assumes a moderate return rate (10%), balancing risk and reward.
    - **Risk-Averse:** Assumes a conservative return rate (6%) for those preferring lower risk.
    """)

    scenario_returns_risk = {
        "Risk-Loving": 0.12,
        "Risk-Neutral": 0.10,
        "Risk-Averse": 0.06
    }
    x_years = np.arange(0, career_years + 1) if career_years > 0 else []
    fig_risk = go.Figure()

    for scenario, ret in scenario_returns_risk.items():
        if len(x_years) > 0:
            savings = [
                current_investments * (1 + ret)**i +
                monthly_contribution * (((1 + ret)**i - 1) / ret) for i in x_years
            ]
            fig_risk.add_trace(go.Scatter(x=x_years, y=savings, name=scenario))

    fig_risk.update_layout(
        title="Risk-Based Retirement Scenarios",
        xaxis_title="Years to Retirement",
        yaxis_title="Projected Savings($)"
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with tab2:
    st.markdown("""
    ### S&P 500-Based Scenario
    What if your client has inherited wealth, or recently sold an asset? This scenario simulates the potential future value of a lump sum investment based on historical volatility and returns of the S&P 500.
    """)

    # Step 1: Acquire historical S&P 500 data
    start = (datetime.datetime.now() - datetime.timedelta(days=365*10))
    end = datetime.datetime.now()
    data = yf.download('^GSPC', start=start, end=end)

    # Debugging: Check Yahoo Finance Data
    st.write("Yahoo Finance Data:")
    st.write(data.tail(2))
    if data.empty:
        st.error("No data retrieved from Yahoo Finance. Please verify the ticker symbol and date range.")
    else:
        # Step 2: Calculate annual returns and volatility
        returns = data['Close'].pct_change().dropna()
        annual_vol = returns.std() * np.sqrt(252)  # Approx. 252 trading days in a year
        annual_mean_return = returns.mean() * 252

        # Debugging: Check volatility and mean return
        st.write(f"Annual Volatility: {annual_vol.values[0]:.1%}")
        st.write(f"Annual Mean Return: {annual_mean_return.values[0]:.1%}")

        # Step 3: Prompt user for lump sum investment
        lump_sum = st.number_input("Enter your lump sum investment amount (USD $):", min_value=0.0, value=10000.0, step=1000.0)

        # Step 4: Initialize simulation parameters
        n_simulations = st.slider("Number of Simulations:", min_value=10, max_value=1000, value=100, step=10)

        # Step 5: Simulate portfolio growth using vectorized operations
        sim_paths = np.zeros((career_years + 1, n_simulations))
        sim_paths[0] = lump_sum

        for t in range(1, career_years + 1):
            random_returns = np.random.normal(annual_mean_return, annual_vol, n_simulations)
            sim_paths[t] = sim_paths[t - 1] * (1 + random_returns)

        simulation_df = pd.DataFrame(sim_paths)

        # Debugging: Check simulation data
        st.write("Simulation DataFrame Summary:")
        st.write(simulation_df.describe())

        # Ensure DataFrame has valid data
        if simulation_df.empty or simulation_df.isna().any().any():
            st.error("Monte Carlo simulation resulted in no valid data. Please verify the inputs.")
        else:
            # Step 6: Visualization
            fig_mc = go.Figure()

            # Add a subset of individual simulations to the plot (e.g., first 10)
            for col in simulation_df.iloc[:, :10].columns:
                fig_mc.add_trace(go.Scatter(
                    x=simulation_df.index,
                    y=simulation_df[col],
                    mode='lines',
                    line=dict(width=0.5),
                    opacity=0.6,
                    showlegend=False
                ))

            # Add the mean path across all simulations
            mean_path = simulation_df.mean(axis=1)
            fig_mc.add_trace(go.Scatter(
                x=simulation_df.index,
                y=mean_path,
                mode='lines',
                line=dict(width=3, color='red'),
                name='Mean Path'
            ))

            # Add confidence intervals (5% and 95% quantiles)
            ci_lower = simulation_df.quantile(0.05, axis=1)
            ci_upper = simulation_df.quantile(0.95, axis=1)

            fig_mc.add_trace(go.Scatter(
                x=simulation_df.index,
                y=ci_lower,
                mode='lines',
                line=dict(color='blue', dash='dash'),
                name='5% CI'
            ))

            fig_mc.add_trace(go.Scatter(
                x=simulation_df.index,
                y=ci_upper,
                mode='lines',
                line=dict(color='blue', dash='dash'),
                name='95% CI'
            ))

            # Configure layout
            fig_mc.update_layout(
                title="Monte Carlo Simulation of Lump Sum Investment",
                xaxis_title="Years",
                yaxis_title="Portfolio Value ($)",
                showlegend=True
            )

            # Display the plot
            st.plotly_chart(fig_mc, use_container_width=True)

            # Step 7: Display summary statistics
            final_values = simulation_df.iloc[-1]
            st.write(f"Mean Portfolio Value at Retirement: ${final_values.mean():.2f}")
            st.write(f"Standard Deviation of Portfolio Value: ${final_values.std():.2f}")

            prob_doubling = (final_values >= 2 * lump_sum).mean() * 100
            st.write(f"Probability of Doubling Your Investment: {prob_doubling:.2f}%")

# Savings vs Inflation Comparison
st.subheader("Savings vs Inflation")
if career_years > 0:
    slider_inflation = st.slider("Select an Inflation Rate(%):", min_value=0.0, max_value=10.0, value=2.0) / 100
    cumulative_inflation = np.cumprod(np.repeat(1 + slider_inflation, career_years))
    inflated_goal = future_savings_goal * cumulative_inflation[-1]

    fig_savings_vs_inflation = go.Figure()
    x_years = np.arange(0, career_years + 1)
    savings_projection = [
        current_investments * (1 + annual_return)**i +
        monthly_contribution * (((1 + annual_return)**i - 1) / annual_return)
        for i in x_years
    ]

    fig_savings_vs_inflation.add_trace(go.Scatter(x=x_years, y=savings_projection, name="Projected Savings"))
    fig_savings_vs_inflation.add_trace(go.Scatter(x=x_years, y=[inflated_goal]*len(x_years), mode='lines', name="Inflation-Adjusted Goal"))
    fig_savings_vs_inflation.update_layout(title="Savings vs Inflation Comparison", xaxis_title="Years to Retirement", yaxis_title="Amount($)")
    st.plotly_chart(fig_savings_vs_inflation, use_container_width=True)
else:
    st.warning("Retirement age must be greater than your current age to calculate savings and inflation projections.")

# Section 7: Assignment Suggestions
st.header("**Student Enhancements**")
st.markdown("""
### Suggested Enhancements:
1. Add additional sliders for dynamic expense or debt repayment adjustments.
2. Include sensitivity analysis for varying inflation or salary growth rates.
3. Include the ability to adjust retirement contributions over time (currently static).
4. Implement tax-advantaged retirement account options (e.g., 401(k), IRA).
5. Build advanced visualizations showing risk-adjusted returns or Monte Carlo simulations.
""")