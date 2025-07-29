# 📊 Evaluating a DCA Strategy with QuantStats Lumi

QuantStats Lumi is a powerful library for portfolio analytics, but when analyzing a Dollar-Cost Averaging (DCA) strategy, some care is needed. Here's how to properly use it.

## 📈 Net Asset Value (NAV) Definition

The NAV $V_t$ for day _t_ is defined as:

$$V_t = \text{cash}_t + \sum_i S_{i,t} \times P_{i,t}$$

where:

- $S_{i,t}$ is the number of shares held for ticker _i_
- $P_{i,t}$ is its adjusted close price

---

## 🧠 Why You Can't Just Use Raw NAV Returns

In a DCA, portfolio value grows not only due to market performance but also because of periodic deposits.

- Raw returns like Vₜ / Vₜ₋₁ mix investment performance with cash inflows.
- Metrics like Sharpe, CAGR, and Drawdown become misleading if based on this.

---

## ✅ Correct Approaches for DCA Analysis

### 1. Time-Weighted Returns (TWR)

TWR isolates **pure investment performance** from the effect of external cash flows, making it the metric of choice for Dollar-Cost Averaging strategies.

Given:

- $V_t$ - portfolio value at the end of day _t_
- $D_t$ - cash deposit on day _t_ (0 on non-deposit days)
- $V_t^{pre}$ - pre-money value: $V_{t-1} + D_t$

The sub-period return is:

$$r_t = \frac{V_t - V_t^{pre}}{V_t^{pre}}$$

TWR is then:

$$\text{TWR} = \prod_t (1 + r_t) - 1$$

**Algorithm:**

1. Calculate NAV $V_t$ with `compute_portfolio_value`.
2. Determine pre-money value $V_t^{pre} = V_{t-1} + D_t$.
3. Evaluate $r_t = (V_t - V_t^{pre}) / V_t^{pre}$.

**Formula:**

- Let:
  - $V_t$ = portfolio value at time t (after deposit and market move)
  - $D_t$ = deposit at time t

- Then:
  - Pre-money value: $V_t^{pre} = V_{t-1} + D_t$
  - Subperiod return: $r_t = (V_t - V_t^{pre}) / V_t^{pre}$

- Chain these returns: $\text{TWR} = \prod(1 + r_t) - 1$

Pass this TWR return series to QuantStats:

```python
qs.reports.metrics(twr_returns.dropna())
```

---

### 2. Money-Weighted Return (IRR / MWR)

Captures the investor’s experience, factoring in the timing and size of cash flows.

Cash flows:

- −Dₜ for each deposit
- +Vₙ as the final portfolio value

Example in code:

```python
cashflows = -deposits.copy()
cashflows.iloc[-1] += nav.iloc[-1]  # add final NAV
irr = qs.stats.irr(cashflows)
```

---

## ⚠️ Don’t Do This

Avoid using this:

```python
returns = nav.pct_change()
qs.reports.metrics(returns)  # ❌ Incorrect if nav includes deposits
```

---

## ✅ Do This Instead

EITHER:

- Build a time-weighted return series (removing deposit effect), or
- Construct a cashflow vector and calculate IRR

Then pass the resulting data into QuantStats for analysis.
