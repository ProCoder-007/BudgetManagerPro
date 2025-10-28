import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import numpy as np
import io

# Page configuration
st.set_page_config(
    page_title="Budget Manager Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    h1 { color: #1f1f1f; padding-bottom: 20px; }
    h2, h3 { color: #2c3e50; }
    .suggestion-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .danger-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state with proper structure
def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = {
            "transactions": [],
            "budgets": {},
            "reminders": [],
            "paid_reminders": [],
            "investments": [],
            "risk_profile": "moderate"
        }
    # Ensure all keys exist
    if "paid_reminders" not in st.session_state.data:
        st.session_state.data["paid_reminders"] = []
    if "investments" not in st.session_state.data:
        st.session_state.data["investments"] = []


init_session_state()


# Helper Functions
def calculate_risk_profile(transactions_df, income, expenses):
    """Calculate user's risk profile"""
    if income == 0:
        return "conservative"
    savings_rate = ((income - expenses) / income) * 100 if income > 0 else 0
    if savings_rate >= 30:
        return "aggressive"
    elif savings_rate >= 15:
        return "moderate"
    else:
        return "conservative"


def generate_investment_suggestions(risk_profile, available_funds, age=30):
    """Generate investment suggestions"""
    suggestions = {
        "conservative": [
            {"name": "High-Yield Savings", "allocation": 40, "expected_return": "3-4%", "risk": "Low",
             "description": "FDIC insured savings account"},
            {"name": "Treasury Bonds", "allocation": 30, "expected_return": "4-5%", "risk": "Low",
             "description": "Government-backed bonds"},
            {"name": "Corporate Bonds", "allocation": 20, "expected_return": "5-6%", "risk": "Low-Medium",
             "description": "Investment-grade corporate bonds"},
            {"name": "Dividend ETF", "allocation": 10, "expected_return": "6-8%", "risk": "Medium",
             "description": "Dividend-paying stocks ETF"}
        ],
        "moderate": [
            {"name": "S&P 500 Index", "allocation": 30, "expected_return": "10-12%", "risk": "Medium",
             "description": "Tracks 500 largest US companies"},
            {"name": "International ETF", "allocation": 20, "expected_return": "8-10%", "risk": "Medium",
             "description": "Global market exposure"},
            {"name": "Bond Index", "allocation": 25, "expected_return": "4-6%", "risk": "Low-Medium",
             "description": "Diversified bond portfolio"},
            {"name": "REIT", "allocation": 15, "expected_return": "8-10%", "risk": "Medium",
             "description": "Real estate investment trust"},
            {"name": "Growth ETF", "allocation": 10, "expected_return": "12-15%", "risk": "Medium-High",
             "description": "High-growth companies"}
        ],
        "aggressive": [
            {"name": "Tech Sector ETF", "allocation": 25, "expected_return": "15-20%", "risk": "High",
             "description": "Technology companies"},
            {"name": "Emerging Markets", "allocation": 20, "expected_return": "12-18%", "risk": "High",
             "description": "Developing economies"},
            {"name": "Small-Cap Growth", "allocation": 20, "expected_return": "14-18%", "risk": "High",
             "description": "Small companies with growth potential"},
            {"name": "S&P 500 Index", "allocation": 20, "expected_return": "10-12%", "risk": "Medium",
             "description": "Core holdings"},
            {"name": "Crypto/Commodities", "allocation": 10, "expected_return": "Variable", "risk": "Very High",
             "description": "Alternative investments"},
            {"name": "Bond Fund", "allocation": 5, "expected_return": "4-6%", "risk": "Low",
             "description": "Stability component"}
        ]
    }

    result = suggestions.get(risk_profile, suggestions["moderate"])
    for item in result:
        item["amount"] = round((item["allocation"] / 100) * available_funds, 2)
    return result


def predict_portfolio_growth(initial, monthly, years, avg_return):
    """Predict portfolio growth"""
    months = years * 12
    monthly_rate = avg_return / 12 / 100
    values = [initial]
    dates = [datetime.now()]
    current = initial

    for month in range(1, months + 1):
        current = (current + monthly) * (1 + monthly_rate)
        values.append(current)
        dates.append(datetime.now() + timedelta(days=30 * month))

    return pd.DataFrame({'date': dates, 'value': values})


def get_market_trends():
    """Enhanced market trends with realistic data"""
    return {
        "Technology": {
            "trend": "Bullish", "confidence": 85,
            "current_index": 15234.56,
            "change_1m": 5.3,
            "change_3m": 12.8,
            "change_1y": 28.4,
            "factors": ["AI adoption accelerating", "Cloud growth strong", "5G expansion"],
            "recommendation": "Strong Buy"
        },
        "Healthcare": {
            "trend": "Bullish", "confidence": 78,
            "current_index": 12456.89,
            "change_1m": 3.2,
            "change_3m": 8.5,
            "change_1y": 15.6,
            "factors": ["Aging demographics", "Biotech innovation", "Digital health growth"],
            "recommendation": "Buy"
        },
        "Energy": {
            "trend": "Neutral", "confidence": 65,
            "current_index": 8932.45,
            "change_1m": 1.8,
            "change_3m": 4.2,
            "change_1y": 9.3,
            "factors": ["Renewable transition", "Oil price stability", "Green tech investment"],
            "recommendation": "Hold"
        },
        "Finance": {
            "trend": "Bullish", "confidence": 72,
            "current_index": 11234.67,
            "change_1m": 4.1,
            "change_3m": 9.7,
            "change_1y": 18.2,
            "factors": ["Interest rate normalization", "Digital banking growth", "Fintech innovation"],
            "recommendation": "Buy"
        },
        "Real Estate": {
            "trend": "Neutral", "confidence": 60,
            "current_index": 9876.54,
            "change_1m": 0.5,
            "change_3m": 2.3,
            "change_1y": 6.8,
            "factors": ["Rate stabilization", "Remote work impact", "Supply constraints"],
            "recommendation": "Hold"
        }
    }


# Sidebar
with st.sidebar:
    st.header("📊 Navigation")
    page = st.radio(
        "Go to",
        ["Dashboard", "Add Transaction", "Set Budgets", "Payment Reminders",
         "AI Investment Advisor", "Portfolio Tracker", "Market Trends", "Reports"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🎨 Quick Stats")

    transactions = st.session_state.data.get("transactions", [])
    if transactions:
        df = pd.DataFrame(transactions)
        total_income = df[df['type'] == 'Income']['amount'].sum()
        total_expenses = df[df['type'] == 'Expense']['amount'].sum()
        balance = total_income - total_expenses

        st.metric("Balance", f"${balance:,.2f}")
        st.metric("Income", f"${total_income:,.2f}")
        st.metric("Expenses", f"${total_expenses:,.2f}")

        investments = st.session_state.data.get("investments", [])
        if investments:
            total_invested = sum(inv['amount'] for inv in investments)
            st.metric("Invested", f"${total_invested:,.2f}")

# Main title
st.title("💰 Budget Manager Pro")
st.markdown("### Your Personal Finance & Investment Dashboard")

# Dashboard Page
if page == "Dashboard":
    st.header("📈 Financial Overview")

    transactions = st.session_state.data.get("transactions", [])

    if not transactions:
        st.info("👋 Welcome! Start by adding your first transaction.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 💵 Add Income")
            st.write("Track salary and income")
        with col2:
            st.markdown("### 💸 Add Expenses")
            st.write("Monitor your spending")
        with col3:
            st.markdown("### 🤖 AI Advisor")
            st.write("Get investment advice")
    else:
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)

        total_income = df[df['type'] == 'Income']['amount'].sum()
        total_expenses = df[df['type'] == 'Expense']['amount'].sum()
        balance = total_income - total_expenses
        this_month_expenses = df[(df['type'] == 'Expense') &
                                 (df['date'].dt.month == datetime.now().month)]['amount'].sum()

        with col1:
            st.metric("💰 Balance", f"${balance:,.2f}")
        with col2:
            st.metric("📥 Income", f"${total_income:,.2f}")
        with col3:
            st.metric("📤 Expenses", f"${total_expenses:,.2f}")
        with col4:
            st.metric("📅 This Month", f"${this_month_expenses:,.2f}")

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💸 Expenses by Category")
            expense_df = df[df['type'] == 'Expense'].groupby('category')['amount'].sum().reset_index()
            if not expense_df.empty:
                fig = px.pie(expense_df, values='amount', names='category', hole=0.4)
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📊 Income vs Expenses")
            monthly_data = df.groupby([df['date'].dt.to_period('M'), 'type'])['amount'].sum().reset_index()
            monthly_data['date'] = monthly_data['date'].astype(str)
            if not monthly_data.empty:
                fig = px.bar(monthly_data, x='date', y='amount', color='type', barmode='group',
                             color_discrete_map={'Income': '#2ecc71', 'Expense': '#e74c3c'})
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

        # Recent Transactions
        st.subheader("📋 Recent Transactions")
        recent_df = df.sort_values('date', ascending=False).head(10)
        display_df = recent_df.copy()
        display_df['amount'] = display_df.apply(
            lambda x: f"${x['amount']:,.2f}" if x['type'] == 'Income' else f"-${x['amount']:,.2f}", axis=1
        )
        st.dataframe(display_df[['date', 'type', 'category', 'description', 'amount']],
                     use_container_width=True, hide_index=True)

# Add Transaction Page
elif page == "Add Transaction":
    st.header("➕ Add New Transaction")

    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox("Type", ["Expense", "Income"])
        categories = ["Food & Dining", "Transportation", "Shopping", "Entertainment",
                      "Bills & Utilities", "Healthcare", "Education", "Other"] if transaction_type == "Expense" \
            else ["Salary", "Freelance", "Investment", "Gift", "Other"]
        category = st.selectbox("Category", categories)
        amount = st.number_input("Amount ($)", min_value=0.01, step=0.01)

    with col2:
        date = st.date_input("Date", datetime.now())
        description = st.text_input("Description")
        notes = st.text_area("Notes", height=100)

    if st.button("💾 Add Transaction", type="primary"):
        if amount > 0:
            transaction = {
                "date": date.strftime("%Y-%m-%d"),
                "type": transaction_type,
                "category": category,
                "amount": float(amount),
                "description": description,
                "notes": notes
            }
            st.session_state.data["transactions"].append(transaction)
            st.success(f"✅ {transaction_type} of ${amount:.2f} added!")
            st.balloons()

# Set Budgets Page
elif page == "Set Budgets":
    st.header("🎯 Budget Management")

    categories = ["Food & Dining", "Transportation", "Shopping", "Entertainment",
                  "Bills & Utilities", "Healthcare", "Education", "Other"]

    budgets = st.session_state.data.get("budgets", {})
    transactions = st.session_state.data.get("transactions", [])
    df = pd.DataFrame(transactions) if transactions else pd.DataFrame()

    col1, col2 = st.columns(2)

    for idx, category in enumerate(categories):
        with col1 if idx % 2 == 0 else col2:
            current_budget = budgets.get(category, 0)

            if not df.empty:
                this_month = df[
                    (df['type'] == 'Expense') &
                    (df['category'] == category) &
                    (pd.to_datetime(df['date']).dt.month == datetime.now().month)
                    ]['amount'].sum()
            else:
                this_month = 0

            with st.expander(f"📁 {category}", expanded=False):
                new_budget = st.number_input(
                    "Monthly Budget ($)",
                    min_value=0.0,
                    value=float(current_budget),
                    step=10.0,
                    key=f"budget_{category}"
                )

                if new_budget > 0:
                    percentage = (this_month / new_budget) * 100
                    st.progress(min(percentage / 100, 1.0))

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Spent", f"${this_month:,.2f}")
                    with col_b:
                        remaining = new_budget - this_month
                        st.metric("Remaining", f"${remaining:,.2f}")

                    # Enhanced budget warnings
                    if percentage > 100:
                        st.markdown(f"""
                        <div class='danger-box'>
                        🚨 <b>BUDGET EXCEEDED!</b><br>
                        Over budget by ${this_month - new_budget:,.2f}
                        </div>
                        """, unsafe_allow_html=True)
                    elif remaining <= 15:
                        st.markdown(f"""
                        <div class='warning-box'>
                        ⚠️ <b>Warning: Only ${remaining:.2f} left!</b><br>
                        You're approaching your budget limit.
                        </div>
                        """, unsafe_allow_html=True)

                if st.button("Save Budget", key=f"save_{category}"):
                    st.session_state.data["budgets"][category] = new_budget
                    st.success(f"✅ Budget saved for {category}")

# Payment Reminders Page
elif page == "Payment Reminders":
    st.header("🔔 Payment Reminders")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Add New Reminder")
        reminder_name = st.text_input("Bill/Payment Name")

        col_a, col_b = st.columns(2)
        with col_a:
            reminder_amount = st.number_input("Amount ($)", min_value=0.01, step=0.01)
            reminder_date = st.date_input("Due Date")
        with col_b:
            reminder_frequency = st.selectbox("Frequency", ["One-time", "Monthly", "Quarterly", "Yearly"])
            reminder_category = st.selectbox("Category",
                                             ["Bills & Utilities", "Subscription", "Insurance", "Loan", "Other"])

        if st.button("➕ Add Reminder", type="primary"):
            if reminder_name and reminder_amount > 0:
                reminder = {
                    "name": reminder_name,
                    "amount": float(reminder_amount),
                    "due_date": reminder_date.strftime("%Y-%m-%d"),
                    "frequency": reminder_frequency,
                    "category": reminder_category,
                    "status": "pending"
                }
                st.session_state.data["reminders"].append(reminder)
                st.success("✅ Reminder added!")

    with col2:
        st.subheader("📊 Upcoming Total")
        reminders = st.session_state.data.get("reminders", [])
        total = sum(r['amount'] for r in reminders if r.get('status') == 'pending')
        st.metric("Total Due", f"${total:,.2f}")

    st.markdown("---")
    st.subheader("📅 Active Reminders")

    reminders = st.session_state.data.get("reminders", [])

    if reminders:
        for idx, reminder in enumerate(reminders):
            due_date = datetime.strptime(reminder['due_date'], "%Y-%m-%d")
            days_until = (due_date - datetime.now()).days

            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

            with col1:
                st.markdown(f"### ⏳ {reminder['name']}")
                st.caption(f"{reminder['category']}")

            with col2:
                st.markdown(f"**${reminder['amount']:,.2f}**")
                st.caption(reminder['frequency'])

            with col3:
                if days_until < 0:
                    st.error(f"Overdue by {abs(days_until)} days")
                elif days_until == 0:
                    st.warning("Due Today!")
                elif days_until <= 7:
                    st.warning(f"Due in {days_until} days")
                else:
                    st.info(f"Due in {days_until} days")

            with col4:
                if st.button("✓ Paid", key=f"paid_{idx}"):
                    paid_reminder = reminder.copy()
                    paid_reminder['paid_date'] = datetime.now().strftime("%Y-%m-%d")
                    paid_reminder['status'] = 'paid'
                    st.session_state.data["paid_reminders"].append(paid_reminder)
                    st.session_state.data["reminders"].pop(idx)
                    st.success("Marked as paid!")
                    st.rerun()

                if st.button("🗑️", key=f"del_{idx}"):
                    st.session_state.data["reminders"].pop(idx)
                    st.success("Reminder deleted!")
                    st.rerun()

            st.markdown("---")
    else:
        st.info("No active reminders. Add one above!")

    # Export paid reminders
    st.markdown("---")
    st.subheader("📥 Export Paid Reminders")
    paid_reminders = st.session_state.data.get("paid_reminders", [])

    if paid_reminders:
        df_paid = pd.DataFrame(paid_reminders)

        # Create Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_paid.to_excel(writer, index=False, sheet_name='Paid Reminders')
        output.seek(0)

        st.download_button(
            label="📥 Download Paid Reminders (Excel)",
            data=output,
            file_name=f"paid_reminders_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.dataframe(df_paid, use_container_width=True, hide_index=True)
    else:
        st.info("No paid reminders yet")

# AI Investment Advisor Page
elif page == "AI Investment Advisor":
    st.header("🤖 AI Investment Advisor")

    transactions = st.session_state.data.get("transactions", [])

    if not transactions:
        st.warning("⚠️ Add transactions first to get personalized recommendations!")
    else:
        df = pd.DataFrame(transactions)
        total_income = df[df['type'] == 'Income']['amount'].sum()
        total_expenses = df[df['type'] == 'Expense']['amount'].sum()
        available = total_income - total_expenses

        st.subheader("📊 Financial Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Income", f"${total_income:,.2f}")
        with col2:
            st.metric("Expenses", f"${total_expenses:,.2f}")
        with col3:
            st.metric("Available", f"${available:,.2f}")

        st.markdown("---")

        auto_risk = calculate_risk_profile(df, total_income, total_expenses)

        col1, col2 = st.columns(2)

        with col1:
            risk_profile = st.selectbox("Risk Tolerance",
                                        ["conservative", "moderate", "aggressive"],
                                        index=["conservative", "moderate", "aggressive"].index(auto_risk)
                                        )
            age = st.slider("Age", 18, 80, 30)
            amount_to_invest = st.number_input("Amount to Invest ($)",
                                               min_value=100.0, max_value=float(available),
                                               value=min(5000.0, float(available)), step=100.0
                                               )

        with col2:
            st.markdown("### Risk Guide")
            st.markdown("**Conservative:** 3-6% return")
            st.markdown("**Moderate:** 7-10% return")
            st.markdown("**Aggressive:** 12-18% return")

        if st.button("🔮 Generate Recommendations", type="primary"):
            suggestions = generate_investment_suggestions(risk_profile, amount_to_invest, age)

            st.success("✅ Analysis Complete!")
            st.markdown("---")

            # Allocation chart
            allocation_df = pd.DataFrame(suggestions)
            fig = px.pie(allocation_df, values='allocation', names='name',
                         title=f"Portfolio Allocation (${amount_to_invest:,.2f})")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("📋 Investment Suggestions")

            for idx, sug in enumerate(suggestions):
                with st.expander(f"💼 {sug['name']} - {sug['allocation']}% (${sug['amount']:,.2f})", expanded=True):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Description:** {sug['description']}")
                        st.markdown(f"**Expected Return:** {sug['expected_return']}")
                        st.markdown(f"**Risk Level:** {sug['risk']}")

                    with col2:
                        st.metric("Amount", f"${sug['amount']:,.2f}")
                        st.metric("Allocation", f"{sug['allocation']}%")

                    if st.button(f"➕ Add to Portfolio", key=f"add_{idx}"):
                        investment = {
                            "name": sug['name'],
                            "amount": sug['amount'],
                            "allocation": sug['allocation'],
                            "risk": sug['risk'],
                            "expected_return": sug['expected_return'],
                            "date_added": datetime.now().strftime("%Y-%m-%d"),
                            "description": sug['description']
                        }
                        st.session_state.data['investments'].append(investment)
                        st.success(f"✅ Added {sug['name']} to portfolio!")
                        st.rerun()

# Portfolio Tracker Page
elif page == "Portfolio Tracker":
    st.header("📊 Portfolio Tracker")

    investments = st.session_state.data.get("investments", [])

    if not investments:
        st.info("📭 No investments yet. Visit AI Advisor to get started!")
    else:
        df_inv = pd.DataFrame(investments)
        total_invested = df_inv['amount'].sum()

        # Simulate current values
        np.random.seed(42)
        df_inv['current_value'] = df_inv['amount'] * np.random.uniform(0.95, 1.15, len(df_inv))
        df_inv['return'] = df_inv['current_value'] - df_inv['amount']
        df_inv['return_pct'] = (df_inv['return'] / df_inv['amount']) * 100

        total_value = df_inv['current_value'].sum()
        total_return = df_inv['return'].sum()
        total_return_pct = (total_return / total_invested) * 100

        st.subheader("📈 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Invested", f"${total_invested:,.2f}")
        with col2:
            st.metric("Current Value", f"${total_value:,.2f}", delta=f"${total_return:,.2f}")
        with col3:
            st.metric("Total Return", f"{total_return_pct:.2f}%")
        with col4:
            diversification = min(len(df_inv) * 15, 100)
            st.metric("Diversification", f"{diversification}/100")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💰 Allocation")
            fig = px.pie(df_inv, values='amount', names='name')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📊 Performance")
            fig = px.bar(df_inv, x='name', y='return_pct', color='return_pct',
                         color_continuous_scale=['red', 'yellow', 'green'])
            fig.update_layout(height=350, showlegend=False,
                              yaxis_title="Return %", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Holdings")

        display_df = df_inv[['name', 'amount', 'current_value', 'return', 'return_pct', 'risk']].copy()
        display_df.columns = ['Investment', 'Invested', 'Current Value', 'Return', 'Return %', 'Risk']
        display_df['Invested'] = display_df['Invested'].apply(lambda x: f"${x:,.2f}")
        display_df['Current Value'] = display_df['Current Value'].apply(lambda x: f"${x:,.2f}")
        display_df['Return'] = display_df['Return'].apply(lambda x: f"${x:,.2f}")
        display_df['Return %'] = display_df['Return %'].apply(lambda x: f"{x:.2f}%")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Growth projection
        st.markdown("---")
        st.subheader("📈 Growth Projection")

        col1, col2 = st.columns(2)
        with col1:
            years = st.slider("Years", 1, 30, 10)
        with col2:
            monthly = st.number_input("Monthly Contribution ($)", min_value=0.0, value=500.0, step=50.0)

        avg_return = 8.0  # Default
        growth_df = predict_portfolio_growth(total_invested, monthly, years, avg_return)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=growth_df['date'], y=growth_df['value'],
                                 mode='lines', fill='tozeroy', line=dict(color='#2ecc71', width=3)))
        fig.update_layout(title=f"Projected Growth ({years} Years)",
                          xaxis_title="Date", yaxis_title="Value ($)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        final_value = growth_df['value'].iloc[-1]
        total_contrib = total_invested + (monthly * 12 * years)
        gains = final_value - total_contrib

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Value", f"${final_value:,.2f}")
        with col2:
            st.metric("Total Contributions", f"${total_contrib:,.2f}")
        with col3:
            st.metric("Total Gains", f"${gains:,.2f}")

# Market Trends Page
elif page == "Market Trends":
    st.header("📊 Market Trends & Analysis")

    trends = get_market_trends()

    st.subheader("🌍 Market Overview")

    bullish = sum(1 for t in trends.values() if t['trend'] == 'Bullish')
    neutral = sum(1 for t in trends.values() if t['trend'] == 'Neutral')
    bearish = sum(1 for t in trends.values() if t['trend'] == 'Bearish')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 Bullish", bullish)
    with col2:
        st.metric("🟡 Neutral", neutral)
    with col3:
        st.metric("🔴 Bearish", bearish)

    st.markdown("---")
    st.subheader("🏢 Sector Analysis")

    for sector, data in trends.items():
        with st.expander(f"📈 {sector} - {data['trend']} ({data['confidence']}% confidence)", expanded=True):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"### Current Index: **${data['current_index']:,.2f}**")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    color_1m = "normal" if data['change_1m'] >= 0 else "inverse"
                    st.metric("1 Month", f"{data['change_1m']:+.1f}%")
                with col_b:
                    st.metric("3 Months", f"{data['change_3m']:+.1f}%")
                with col_c:
                    st.metric("1 Year", f"{data['change_1y']:+.1f}%")

                st.markdown("**Key Factors:**")
                for factor in data['factors']:
                    st.markdown(f"• {factor}")

            with col2:
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=data['confidence'],
                    title={'text': "Confidence"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ecc71" if data['trend'] == 'Bullish'
                        else "#f39c12" if data['trend'] == 'Neutral' else "#e74c3c"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "darkgray"}
                        ]
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                rec_color = "#2ecc71" if 'Buy' in data['recommendation'] else "#f39c12" if data[
                                                                                               'recommendation'] == 'Hold' else "#e74c3c"
                st.markdown(
                    f"<div style='background-color: {rec_color}; padding: 10px; border-radius: 10px; "
                    f"text-align: center; color: white; font-weight: bold;'>"
                    f"Recommendation: {data['recommendation']}</div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.subheader("📈 Historical Performance (Simulated)")

    # Generate realistic market data
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')

    # More realistic market simulation
    np.random.seed(42)
    tech_base = 1000
    healthcare_base = 1000
    sp500_base = 1000

    tech_index = []
    healthcare_index = []
    sp500_index = []

    for i in range(365):
        tech_base *= (1 + np.random.normal(0.0008, 0.015))
        healthcare_base *= (1 + np.random.normal(0.0004, 0.010))
        sp500_base *= (1 + np.random.normal(0.0003, 0.008))

        tech_index.append(tech_base)
        healthcare_index.append(healthcare_base)
        sp500_index.append(sp500_base)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=tech_index, name='Technology',
                             line=dict(color='#667eea', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=healthcare_index, name='Healthcare',
                             line=dict(color='#2ecc71', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=sp500_index, name='S&P 500',
                             line=dict(color='#f39c12', width=2)))

    fig.update_layout(
        title="1-Year Sector Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Index Value",
        height=450,
        hovermode='x unified',
        yaxis=dict(tickformat=",.0f")
    )

    st.plotly_chart(fig, use_container_width=True)

    # Market predictions
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='suggestion-box'>
        <h4>📈 Next Quarter Outlook</h4>
        <p><b>Positive Trends:</b></p>
        <ul>
            <li>Tech sector momentum strong (+15-20% YoY)</li>
            <li>Healthcare innovation accelerating</li>
            <li>Financial sector recovering</li>
        </ul>
        <p><b>Expected Movement:</b> Moderately Bullish (+5-8%)</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='warning-box'>
        <h4>⚠️ Risk Factors</h4>
        <ul>
            <li>Interest rate volatility</li>
            <li>Geopolitical tensions</li>
            <li>Inflation concerns</li>
            <li>Supply chain issues</li>
        </ul>
        <p><b>Volatility:</b> Moderate (Expected range: ±10%)</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "💡 **Note:** Market data is simulated for demonstration. For live data, integrate with financial APIs like Alpha Vantage or Yahoo Finance.")

# Reports Page
elif page == "Reports":
    st.header("📊 Financial Reports")

    transactions = st.session_state.data.get("transactions", [])

    if not transactions:
        st.info("No transaction data available")
    else:
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", df['date'].min().date())
        with col2:
            end_date = st.date_input("End Date", df['date'].max().date())

        filtered_df = df[(df['date'] >= pd.Timestamp(start_date)) &
                         (df['date'] <= pd.Timestamp(end_date))]

        st.subheader("📈 Period Summary")
        col1, col2, col3 = st.columns(3)

        period_income = filtered_df[filtered_df['type'] == 'Income']['amount'].sum()
        period_expenses = filtered_df[filtered_df['type'] == 'Expense']['amount'].sum()
        period_balance = period_income - period_expenses

        with col1:
            st.metric("Income", f"${period_income:,.2f}")
        with col2:
            st.metric("Expenses", f"${period_expenses:,.2f}")
        with col3:
            st.metric("Balance", f"${period_balance:,.2f}")

        st.markdown("---")
        st.subheader("💰 Category Breakdown")

        col1, col2 = st.columns(2)

        with col1:
            expense_cat = filtered_df[filtered_df['type'] == 'Expense'].groupby('category')[
                'amount'].sum().reset_index()
            if not expense_cat.empty:
                fig = px.bar(expense_cat.sort_values('amount', ascending=True),
                             y='category', x='amount', orientation='h',
                             color='amount', color_continuous_scale='Reds',
                             title="Expenses by Category")
                fig.update_layout(height=400, showlegend=False,
                                  yaxis_title="", xaxis_title="Amount ($)",
                                  xaxis=dict(tickformat="$,.0f"))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            income_cat = filtered_df[filtered_df['type'] == 'Income'].groupby('category')['amount'].sum().reset_index()
            if not income_cat.empty:
                fig = px.bar(income_cat.sort_values('amount', ascending=True),
                             y='category', x='amount', orientation='h',
                             color='amount', color_continuous_scale='Greens',
                             title="Income by Category")
                fig.update_layout(height=400, showlegend=False,
                                  yaxis_title="", xaxis_title="Amount ($)",
                                  xaxis=dict(tickformat="$,.0f"))
                st.plotly_chart(fig, use_container_width=True)

        # Monthly trend
        st.markdown("---")
        st.subheader("📈 Monthly Trend")

        monthly_summary = filtered_df.groupby([filtered_df['date'].dt.to_period('M'), 'type'])[
            'amount'].sum().reset_index()
        monthly_summary['date'] = monthly_summary['date'].astype(str)

        if not monthly_summary.empty:
            fig = px.line(monthly_summary, x='date', y='amount', color='type',
                          markers=True, line_shape='spline',
                          color_discrete_map={'Income': '#2ecc71', 'Expense': '#e74c3c'})
            fig.update_layout(height=350, xaxis_title="Month", yaxis_title="Amount ($)",
                              yaxis=dict(tickformat="$,.0f"))
            st.plotly_chart(fig, use_container_width=True)

        # Export options
        st.markdown("---")
        st.subheader("📥 Export Data")

        col1, col2 = st.columns(2)

        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"transactions_{start_date}_{end_date}.csv",
                mime="text/csv"
            )

        with col2:
            # Excel export
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Transactions')

                # Add summary sheet
                summary_df = pd.DataFrame({
                    'Metric': ['Total Income', 'Total Expenses', 'Net Balance'],
                    'Amount': [period_income, period_expenses, period_balance]
                })
                summary_df.to_excel(writer, index=False, sheet_name='Summary')
            output.seek(0)

            st.download_button(
                label="📥 Download Excel",
                data=output,
                file_name=f"report_{start_date}_{end_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # Detailed transactions table
        st.markdown("---")
        st.subheader("📋 All Transactions")

        display_df = filtered_df.copy()
        display_df['amount'] = display_df.apply(
            lambda x: f"${x['amount']:,.2f}" if x['type'] == 'Income' else f"-${x['amount']:,.2f}",
            axis=1
        )
        st.dataframe(display_df[['date', 'type', 'category', 'description', 'amount']],
                     use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>Budget Manager Pro with AI Investment Advisor</b></p>
    <p style='font-size: 12px;'>Investment suggestions are for educational purposes. Consult a financial advisor for personalized advice.</p>
    <p style='font-size: 12px;'>Market data is simulated. For real-time data, integrate APIs like Alpha Vantage or Yahoo Finance.</p>
</div>
""", unsafe_allow_html=True)