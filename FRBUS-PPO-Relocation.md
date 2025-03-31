# Fiscal Policy Strategy: Integrating FRB/US with Reinforcement Learning and Active Relocation

## Abstract

Policymakers need advanced tools to optimize fiscal policies that stabilize economies and maintain macroeconomic targets. While the Federal Reserve's FRB/US model provides sophisticated forecasting capabilities, its reliance on predefined scenarios limits exploration of the full policy space. This paper integrates FRB/US with Proximal Policy Optimization (PPO) reinforcement learning enhanced by an active relocation mechanism. Our framework uses the FRB/US model as a simulation environment where a PPO agent dynamically adjusts fiscal instruments while strategically relocating to different economic states during training—significantly improving exploration efficiency and convergence speed. Through systematic search of high-dimensional policy spaces, the agent discovers adaptive fiscal strategies that effectively respond to various economic conditions. Results demonstrate that this approach enhances macroeconomic stabilization under diverse scenarios, including historical comparison (1975-2024) and extreme policy situations such as a hypothetical 50% tariff shock (2025-2075), providing policymakers with a powerful decision-support tool that balances multiple objectives including inflation control, employment targets, and fiscal sustainability.

## 1. Introduction

Macroeconomic policymaking requires balancing multiple competing objectives while navigating complex, uncertain environments. Fiscal policy design is particularly challenging as policymakers must simultaneously consider growth, employment, inflation, debt sustainability, and distributional impacts. Traditional approaches to fiscal policy analysis rely on large-scale macroeconomic models with manually specified policy rules and scenarios, potentially missing optimal policy configurations in the vast multidimensional policy space.

The Federal Reserve Board's FRB/US model represents the state-of-the-art in structural macroeconomic modeling, with a rich history of guiding U.S. monetary and fiscal policy decisions. It features sophisticated descriptions of household, firm, and financial market behavior, capturing both short-run dynamics and long-run equilibrium properties. However, the conventional application of such models typically involves analyzing a small set of predefined policy alternatives, rather than systematically exploring the full range of possible policy responses across diverse economic conditions.

Recent advances in reinforcement learning (RL) offer promising new approaches for policy optimization in complex environments. RL methods are designed for sequential decision-making under uncertainty, learning through repeated interaction with an environment to discover policies that maximize cumulative rewards. Among these methods, Proximal Policy Optimization (PPO) has emerged as particularly effective for high-dimensional control problems due to its sample efficiency and stability.

This paper introduces a novel framework that integrates the FRB/US macroeconomic model with PPO reinforcement learning enhanced by an active relocation mechanism. The key innovation lies in combining the structural economic relationships embedded in FRB/US with the adaptive learning capabilities of reinforcement learning, creating a powerful tool for fiscal policy design that can discover non-obvious policy strategies while respecting economic constraints.

Our approach offers several advantages over traditional methods:

1. **Systematic policy space exploration**: Rather than evaluating a handful of predefined policy alternatives, our framework systematically explores high-dimensional fiscal policy configurations, identifying strategies that might be overlooked in conventional analysis.

2. **Adaptive policy responses**: The learned policies adapt to changing economic conditions, providing state-contingent guidance rather than fixed rules.

3. **Balance of multiple objectives**: The reward function can incorporate multiple policy objectives, allowing for explicit tradeoffs between competing goals such as growth, inflation, and fiscal sustainability.

4. **Enhanced learning efficiency**: Our active relocation mechanism enables the agent to strategically navigate the economic state space during training, focusing on informative or challenging scenarios to accelerate convergence.

We evaluate our framework through extensive historical comparison spanning five decades (1975-2024) and through a hypothetical long-term analysis of a severe trade policy shock (50% import tariffs) extending to 2075. The results demonstrate that our RL-enhanced approach consistently outperforms conventional policy approaches across multiple economic indicators while balancing competing objectives more effectively.

The remainder of this paper is organized as follows: Section 2 reviews the literature on macroeconomic modeling and reinforcement learning applications. Section 3 outlines the foundations of reinforcement learning for economic policy. Section 4 presents our integrated framework combining FRB/US with PPO and active relocation. Section 5 describes our experimental design and methodology. Section 6 presents empirical results and economic analysis. Section 7 concludes with policy implications and directions for future research.

## 2. Literature Review and Model Background

### 2.1 Historical Evolution of Macroeconomic Models

Macroeconomic modeling has evolved substantially since the pioneering work of Tinbergen and Klein in the mid-20th century. Early large-scale econometric models provided valuable insights but faced criticism for their lack of microeconomic foundations and vulnerability to the Lucas critique. The 1970s and 1980s saw the development of rational expectations models, including Real Business Cycle (RBC) frameworks and, later, New Keynesian models incorporating nominal rigidities.

Modern structural macroeconomic models attempt to bridge theoretical consistency with empirical realism. The Federal Reserve's FRB/US model exemplifies this approach, combining forward-looking behavior with empirically estimated dynamics. As described by Brayton and Tinsley (1996), FRB/US represents "a large-scale quarterly econometric model of the U.S. economy" where "most behavioral equations are based on specifications of optimizing behavior containing explicit expectations of firms, households, and financial markets."

### 2.2 Overview of the FRB/US Model

The FRB/US model serves as the Federal Reserve's primary workhorse for policy analysis and forecasting. It integrates multiple sectors with rich behavioral descriptions:

**Household Sector**: The model captures consumption decisions through lifecycle planning with adjustment frictions. As noted in the model documentation, "aggregate consumption is derived from the lifecycle model and, thus, depends on the current value of tangible assets and current and expected future values of household income." This formulation allows households to respond to anticipated policy changes while exhibiting realistic adjustment patterns.

**Firm Sector**: Firms make investment, employment, pricing, and production decisions based on profit maximization under imperfect competition. The model incorporates significant real and nominal rigidities, with "adjustment dynamics... estimated to be most rapid for inventories and labor hours and slowest for wages and investment."

**Financial Markets**: Asset prices, including bond yields and equity valuations, are determined through arbitrage relationships. The model captures how "key transmission channels operate through medium- and long-term interest rates directly in equations for investment in producers' durable equipment, residential construction, and consumer durables, and indirectly through effects of the value of the stock market."

**Government Sector**: Fiscal policy variables include government spending, tax rates, and transfer payments, with endogenous feedback from economic conditions to tax revenues and countercyclical spending programs.

**Expectations Formation**: A distinctive feature of FRB/US is its flexible treatment of expectations, accommodating both model-consistent (rational) expectations and limited-information alternatives. This flexibility is crucial for our integrated approach, as it allows us to model adaptive learning within a structurally consistent framework.

### 2.3 Limitations of Traditional Policy Analysis

Despite its sophistication, traditional analysis using FRB/US and similar models faces several limitations:

1. **Limited Exploration**: Conventional analysis typically examines a small set of policy scenarios rather than systematically exploring the full space of possible policy configurations.

2. **Static Design**: Traditional policy rules are often fixed rather than state-contingent, limiting their adaptability to changing economic conditions.

3. **Constrained Optimization**: While FRB/US can evaluate policy effectiveness, it doesn't natively provide optimal policy solutions across multiple competing objectives.

4. **Episodic Analysis**: Traditional applications often focus on isolated episodes rather than consistent policy strategies across diverse economic conditions.

These limitations motivate our integrated approach combining FRB/US with reinforcement learning, which enables systematic policy optimization while maintaining the structural economic relationships that give FRB/US its predictive power.

## 3. Foundations of Reinforcement Learning in Macroeconomic Policy

### 3.1 Basic Concepts in Reinforcement Learning

Reinforcement learning provides a framework for learning optimal decision strategies through interaction with an environment. The key components of RL include:

- **State (s)**: The observable information about the environment, in our case, a vector of macroeconomic variables.
- **Action (a)**: The decision made by the agent, corresponding to fiscal policy adjustments.
- **Reward (r)**: A scalar signal indicating the desirability of outcomes, designed to reflect policy objectives.
- **Policy (π)**: A mapping from states to actions, representing the decision strategy.
- **Value Function (V)**: The expected cumulative reward when following a policy from a given state.

In the context of fiscal policy, the state might include variables such as GDP, inflation, unemployment, and the debt-to-GDP ratio. Actions would correspond to adjustments in fiscal instruments like tax rates and government expenditures. The reward function would incorporate policy objectives such as minimizing deviations from target inflation and unemployment while maintaining fiscal sustainability.

### 3.2 Proximal Policy Optimization (PPO)

Proximal Policy Optimization, introduced by Schulman et al. (2017), represents a significant advancement in policy gradient methods for reinforcement learning. PPO addresses key challenges in policy optimization:

1. **Sample Efficiency**: PPO makes multiple optimization passes over the same trajectory data, improving data efficiency.

2. **Stability**: By using a clipped objective function, PPO limits the size of policy updates, preventing destructively large changes:

$$L^{CLIP}(\theta) = \hat{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio between the new and old policies, $\hat{A}_t$ is the estimated advantage, and $\epsilon$ is a hyperparameter (typically 0.1 or 0.2).

3. **Actor-Critic Architecture**: PPO employs separate networks for the policy (actor) and value function (critic), enabling efficient estimation of the advantage function.

These properties make PPO particularly well-suited for fiscal policy optimization, where stability and sample efficiency are crucial given the complexity of macroeconomic dynamics and the computational cost of simulating the FRB/US model.

### 3.3 Active Relocation and Uncertainty Mechanisms

A key innovation in our approach is the integration of active relocation mechanisms with PPO. Active relocation allows the agent to strategically "jump" to different economic states during training rather than progressing sequentially through states determined by the environment dynamics. This capability is particularly valuable for macroeconomic policy learning for several reasons:

1. **Efficient Exploration**: Economic crises and other challenging policy scenarios may be rare in sequential simulation but critically important for policy learning. Active relocation enables focused learning on these important but infrequent states.

2. **Diverse Experience**: By strategically sampling from different economic conditions, the agent develops more robust policies that generalize across diverse scenarios.

3. **Prioritized Learning**: The agent can focus on states where policy uncertainty is highest, accelerating convergence to optimal policies.

Building on concepts introduced by Mihalkova and Mooney (2006), we implement two relocation strategies:

1. **"Bored" Relocation**: Triggers relocation when learning progress stagnates, with probability proportional to the stability of value estimates.

2. **"Trouble" Relocation**: Increases relocation probability when entering states with declining value, focusing learning on challenging economic conditions.

When relocation is triggered, the agent strategically selects destination states based on policy uncertainty, measured by the standard deviation of the policy distribution. This focuses learning on states where the optimal policy is least certain, significantly enhancing learning efficiency.

## 4. Proposed Framework: Integrating FRB/US and PPO with Active Relocation

### 4.1 Conceptual Architecture

Our framework integrates the FRB/US macroeconomic model with a PPO reinforcement learning agent enhanced by active relocation capabilities. The overall architecture consists of three primary components:

1. **FRB/US Simulation Environment**: Serves as the environment with which the agent interacts, evolving according to the structural relationships embedded in FRB/US and responding to the agent's policy actions.

2. **PPO Agent**: Observes economic states, selects fiscal policy adjustments, and learns from feedback through the PPO algorithm.

3. **Active Relocation Mechanism**: Enhances exploration by strategically repositioning the agent within the economic state space during training.

In this closed-loop system, the agent's policy decisions propagate through the FRB/US model to influence economic outcomes, generating rewards that guide policy optimization. The active relocation mechanism allows the agent to focus learning on informative or challenging economic scenarios, significantly improving training efficiency.

### 4.2 State Space Design

The state representation comprises a comprehensive vector of macroeconomic variables from the FRB/US model. For a given quarter $t$, the state vector $s_t$ includes:

$$s_t = [\pi_t, u_t, g_t, r_t, d_t, tr_t, ge_t, tx_p, tx_c, ex_t, im_t, fs_t,...]$$

where:
- $\pi_t$ represents inflation (PCPI)
- $u_t$ denotes unemployment rate (LUR)
- $g_t$ captures GDP growth (HGGDP)
- $r_t$ indicates interest rates (RFF)
- $d_t$ is the debt-to-GDP ratio
- $tr_t$ represents transfer payments ratio
- $ge_t$ denotes government expenditures
- $tx_p$ and $tx_c$ are personal and corporate tax rates
- $ex_t$ and $im_t$ represent exports and imports
- $fs_t$ captures the federal surplus

This comprehensive state representation enables the agent to perceive the economy's current condition and recent trajectory, facilitating informed fiscal policy decisions. The FRB/US model provides a rich state space with over 60 stochastic equations, 320 identities, and 125 exogenous variables, giving the agent access to detailed information about economic conditions.

### 4.3 Action Space and Policy Instruments

The agent controls four key fiscal policy instruments, adjusting them relative to their previous values. For each quarter $t$, the action vector $a_t$ is defined as:

$$a_t = [\Delta tr_t, \Delta ge_t, \Delta tx_{p,t}, \Delta tx_{c,t}]$$

where:
- $\Delta tr_t$ represents the adjustment to transfer payments ratio, bounded by $[-0.1, 0.1]$
- $\Delta ge_t$ denotes the change in government expenditures, bounded by $[-0.25, 0.25]$ billion dollars
- $\Delta tx_{p,t}$ captures the adjustment to personal tax rates, bounded by $[-0.1, 0.1]$
- $\Delta tx_{c,t}$ represents the change in corporate tax rates, bounded by $[-0.1, 0.1]$

The PPO agent outputs a stochastic policy $\pi_\theta(a|s)$ modeled as a multivariate Gaussian distribution with state-dependent means $\mu_\theta(s)$ and standard deviations $\sigma_\theta(s)$:

$$\pi_\theta(a|s) = \mathcal{N}(a|\mu_\theta(s), \sigma_\theta(s)^2)$$

This stochastic policy allows for exploration during training while gradually converging toward deterministic policy choices as uncertainty decreases.

### 4.4 Reward Function Formulation

The reward function quantifies the desirability of economic outcomes resulting from policy decisions. Our formulation balances multiple objectives including price stability, employment, growth, and fiscal sustainability:

$$r_t = -\alpha_1 \delta_{\pi}(t) - \alpha_2 \delta_{u}(t) + \alpha_3 \delta_{g}(t) - \alpha_4 \delta_{vol}(t) + \alpha_5 \gamma(t) - \alpha_6 \delta_{m}(t) - \alpha_7 \delta_{d}(t)$$

where:
- $\delta_{\pi}(t) = \text{clamp}(|\pi_t - \pi^*|/\pi_{max}, -1, 1)$ represents normalized inflation deviation
- $\delta_{u}(t) = \text{clamp}(|u_t - u^*|/u_{max}, -1, 1)$ captures unemployment deviation
- $\delta_{g}(t) = \text{clamp}((g_t - g^*)/g_{max}, -1, 1)$ reflects GDP growth deviation
- $\delta_{vol}(t)$ measures economic volatility across key indicators
- $\gamma(t)$ compares current growth to historical averages
- $\delta_{m}(t)$ represents a normalized "misery index" combining inflation and unemployment
- $\delta_{d}(t) = \text{clamp}((d_t - d^*)/d_{max}, 0, 1)$ captures debt sustainability concerns

The coefficients $\alpha_1$ through $\alpha_7$ weight these components according to policy priorities, while target values $\pi^*, u^*, g^*, d^*$ represent desired economic outcomes.

### 4.5 Active Relocation Mechanism

The active relocation mechanism enhances learning efficiency by allowing the agent to strategically navigate the economic state space during training. The framework implements two relocation strategies:

1. The "Bored" method triggers relocation when learning progress stagnates, with probability:

$$\pi = \frac{0.5 \cdot e^{-\phi \cdot \frac{|V(s_t) - V(s_{t-1})|}{V_{max}}}}{1 + c}$$

where $V(s_t)$ is the estimated value of the current state, $V_{max}$ tracks the maximum observed value difference, $\phi$ is a scaling parameter, and $c$ represents the relocation cost.

2. The "Trouble" method increases relocation probability when entering unfavorable states:

$$\pi_t = \begin{cases}
\frac{\pi_{t-1} + \tau \cdot (V(s_{t-1}) - V(s_t))}{1 + c} & \text{if } V(s_t) < V(s_{t-1}) \\
\frac{\epsilon}{1 + c} & \text{otherwise}
\end{cases}$$

where $\tau$ controls the adaptation rate and $\epsilon$ establishes a baseline probability.

When relocation is triggered, the agent selects a destination state $s^*$ from previously visited states based on policy uncertainty:

$$s^* = \arg\max \frac{1}{n}\sum_{i=1}^{n}\sigma_i(s) \text{ with } s \in S_{visited}$$

where $\sigma_i(s)$ represents the standard deviation of the policy for the $i$-th action dimension, capturing the agent's uncertainty about optimal fiscal actions.

## 5. Experimental Design and Methodology

### 5.1 Implementation and Training

We implemented our framework using the official FRB/US model as the simulation environment, with the PPO algorithm based on the architecture described by Schulman et al. (2017). The neural network policy (actor) and value function (critic) each consisted of fully connected networks with three hidden layers of 128 units, layer normalization, and ReLU activations.

The actor network outputs both the mean and standard deviation for each action dimension, allowing for stochastic policy sampling, while the critic network estimates the state value function. Both networks are independently parameterized but trained simultaneously.

Training was conducted using the following hyperparameters:
- Learning rate: 0.0003
- Discount factor (γ): 0.99
- GAE parameter (λ): 0.95
- Number of epochs per update (K_epochs): 10
- Clipping parameter (ε): 0.2
- Horizon: 512
- Minibatch size: 4096
- Number of parallel environments: 32 (for standard simulations), 128 (for longer-horizon projections)
- Value function coefficient: 1
- Entropy coefficient: 0.01


We implemented our framework using the official FRB/US model as the simulation environment, with training data derived from two primary sources: historical macroeconomic time series (HISTDATA.TXT) covering 1975-2024 with quarterly observations of over 900 economic variables, and baseline long-term projections (LONGBASE.TXT) extending to 2075. These datasets, maintained by the Federal Reserve, contain comprehensive information on output, inflation, employment, interest rates, fiscal variables, and international trade. The historical data serves dual purposes: providing initial conditions for simulations and enabling calculation of model residuals, which capture the stochastic components of economic relationships. For counterfactual simulations, we apply these empirically-derived residuals to ensure realistic economic dynamics beyond the baseline projections.

The neural network architecture powering our PPO agent features three hidden layers of 128 units each with layer normalization and ReLU activations throughout. This architecture was specifically designed to capture complex economic relationships across hundreds of variables. The actor network outputs both means and standard deviations for each action dimension, enabling stochastic policy sampling, while the critic network estimates state values to guide the learning process.

Training proceeds through sequential quarterly economic decisions where the agent interacts with the FRB/US model, observing economic states, taking fiscal policy actions, and receiving feedback. For historical comparisons (1975-2024), we used the actual historical data as the starting point, allowing the agent to learn policy responses to historical economic conditions. For the tariff shock analysis (2025-2075), we initialized from the final historical quarter and projected forward using the FRB/US model's structural relationships. In both scenarios, the agent progressively learns through repeated simulation replications, with each replication comprising a complete economic episode from start to end dates.

Our implementation leverages a robust reinforcement learning configuration with a learning rate of 0.0003, discount factor (γ) of 0.99 for multi-quarter planning, and PPO clipping parameter (ε) of 0.2 to prevent destructively large policy updates. We employed Generalized Advantage Estimation with λ=0.95 to balance bias and variance in advantage calculations. For training efficiency, we conducted multiple sequential simulation replications rather than parallel environments, collecting experiences through complete economic episodes before performing policy updates. After each full simulation cycle spanning the designated time period, the PPO algorithm processes the collected experiences through 10 epochs of optimization to refine the policy, with checkpoint saving based on performance metrics.

The active relocation mechanism enhances training efficiency by enabling strategic exploration of the economic state space. When triggered through either the "Bored" or "InTrouble" methods, the agent can jump to previously visited economic states with high uncertainty, focusing learning on challenging scenarios while gradually increasing the cost of relocation to ensure stable long-term behavior. All simulations were implemented in PyTorch and conducted with a fixed random seed for reproducibility.
 

### 5.2 Historical Comparison (1975-2024)

To evaluate our framework's performance against historical policy decisions, we conducted a comparative analysis covering five decades of U.S. economic history from 1975 to 2024. This period encompasses diverse economic conditions including:

- Stagflation and Volcker's disinflation (1975-1984)
- The Great Moderation (1985-1994)
- Tech boom, bust, and recovery (1995-2004)
- Housing boom, Great Recession, and slow recovery (2005-2014)
- Late expansion, pandemic shock, and inflation surge (2015-2024)

For each period, we compared three scenarios:
1. **Historical Data**: Actual economic outcomes with historical policy decisions
2. **FRB/US Simulation**: Counterfactual simulation using traditional FRB/US with baseline policy rules
3. **AI Decision Makers/FRB/US**: Our integrated approach with PPO-optimized fiscal policy

This comprehensive comparison allowed us to assess how our RL-enhanced approach would have performed relative to historical policy decisions across diverse economic conditions.

### 5.3 Tariff Shock Analysis (2025-2075)

To demonstrate our framework's capability for long-term policy analysis under extreme scenarios, we simulated the effects of a hypothetical 50% import tariff implemented from 2025 to 2075. This severe trade policy shock creates a challenging environment for policy optimization, requiring delicate balancing of inflation, employment, growth, and fiscal sustainability objectives.

For this analysis, we compared four scenarios:
1. **AI Decision Makers/FRB/US with Tariff**: Our integrated approach responding to the tariff shock
2. **AI Decision Makers/FRB/US without Tariff**: Baseline scenario with our approach but no tariff
3. **FRB/US-Based Simulation with Tariff**: Traditional FRB/US simulation with the tariff shock
4. **FRB/US-Based Simulation without Tariff**: Traditional FRB/US baseline without tariff

This comparative design allows us to isolate both the effects of the tariff shock and the differential performance of our RL-enhanced approach relative to traditional policy frameworks.

### 5.4 Evaluation Metrics

We evaluated policy performance using a comprehensive set of macroeconomic indicators:

1. **Core Macroeconomic Performance**:
   - GDP growth and level
   - Unemployment rate
   - Inflation rate
   - Interest rates

2. **Fiscal Outcomes**:
   - Federal budget balance
   - Debt-to-GDP ratio
   - Tax revenue
   - Transfer payments

3. **External Sector**:
   - Export and import volumes
   - Trade balance

4. **Policy Responsiveness**:
   - Tax rate adjustments
   - Government expenditure changes
   - Transfer payment adjustments

These metrics enable a multidimensional assessment of policy effectiveness across different economic objectives and time horizons.

## 6. Results and Analysis

### 6.1 Historical Comparison (1975-2024)

Our historical comparison reveals that the AI Decision Makers/FRB/US approach consistently outperforms both historical policy decisions and traditional FRB/US simulations across multiple dimensions.

#### 6.1.1 Macroeconomic Stability

The AI approach demonstrates superior countercyclical policy management throughout the five decades. During recessions (1980-1982, 1990-1991, 2001, 2008-2009, 2020), the AI model consistently shows:

- **Less severe unemployment peaks**: During the 1982 recession, historical unemployment reached 10.9% while the AI model peaked at 9.9%. Similar outperformance occurs during the Great Recession, with unemployment 0.6 percentage points lower in the AI model.

- **Stronger post-recession recoveries**: Following the 1990-1991 and 2001 recessions, the AI model facilitated faster returns to potential output, with GDP approximately 1% higher than historical levels by 1994 and 2004, respectively.

- **More effective inflation management**: During the Great Inflation (1975-1984), the AI model achieved lower peak inflation while still maintaining better unemployment outcomes. Similarly, during the 2021-2022 inflation surge, the AI model showed more moderate inflation with comparable employment performance.

#### 6.1.2 Fiscal Policy Responsiveness

The AI approach demonstrates distinctly different fiscal policy patterns compared to historical experience:

- **More countercyclical tax policy**: The AI model implements more responsive tax adjustments, with larger tax reductions during recessions (beginning 1-2 quarters earlier than historical policy) and tax increases during expansions to rebuild fiscal capacity.

- **Strategic transfer payments**: Government transfer payments in the AI model show stronger automatic stabilizer effects, with higher payments during downturns but more disciplined growth during expansions.

- **Better fiscal sustainability**: Despite more aggressive countercyclical policy, the AI model achieves better long-run fiscal outcomes, with federal budget surpluses approximately 15-20% larger during expansions (particularly evident in the late 1990s).

#### 6.1.3 Monetary-Fiscal Coordination

A distinctive feature of the AI approach is more effective coordination between fiscal and monetary policies:

- **Complementary rather than offsetting actions**: During disinflation periods, the AI model coordinates tighter monetary policy with fiscal restraint, enhancing effectiveness. During recessions, it balances monetary accommodation with fiscal support.

- **More dynamic interest rate adjustments**: The AI model shows both more aggressive initial rate changes when conditions warrant and faster normalization once objectives are achieved, avoiding the policy inertia that characterized some historical episodes.

- **Enhanced policy credibility**: By maintaining more consistent long-term policy objectives while flexibly responding to short-term conditions, the AI approach achieves better anchoring of expectations, evidenced by smaller term premiums in long-term interest rates.

### 6.2 Tariff Shock Analysis (2025-2075)

Our analysis of the hypothetical 50% tariff shock provides insights into both the economic effects of extreme trade policy changes and the differential performance of our RL-enhanced approach in managing such shocks.

#### 6.2.1 Immediate Tariff Impact (2025-2035)

The tariff implementation creates significant economic disruption across all scenarios:

- **Severe growth volatility**: The AI Decision Makers/FRB/US with tariff scenario shows dramatic fluctuations, including a 10.7% growth spike in 2026Q3 followed by a -6.8% contraction in 2026Q4, indicating major resource reallocation.

- **Price level adjustment**: All tariff scenarios show substantial one-time price level increases as import costs rise, with the consumer price index jumping approximately 5-6 percentage points.

- **Trade volume contraction**: Import volumes decline by 20-25% within two years of tariff implementation, while exports fall by 10-15% due to global feedback effects and exchange rate adjustments.

- **Initial fiscal deterioration**: Despite generating tariff revenue, the Federal budget balance worsens in all tariff scenarios, with the AI Decision Makers model showing deficit spikes to over $14 trillion during the adjustment period compared to $1.5-2 trillion in non-tariff scenarios.

#### 6.2.2 Medium-Term Adaptation (2035-2055)

As the economy adapts to the permanent tariff regime, several patterns emerge:

- **Growth convergence with efficiency loss**: Growth rates in tariff scenarios gradually converge toward non-tariff rates (around 1.7-1.8% annually) as the economy adapts, but at a permanently lower output level. By 2055, real GDP in tariff scenarios is approximately 6-7% below potential compared to non-tariff scenarios.

- **Fiscal burden accumulation**: Despite similar growth rates, fiscal positions continue to deteriorate in tariff scenarios, with debt-to-GDP ratios reaching 89-90% versus 81-82% in non-tariff scenarios by 2055.

- **Higher transfer dependency**: Government transfer payments remain 8-10% higher in tariff scenarios despite the smaller economy, indicating structural dependency created by the trade restriction.

- **Trade deficit reduction with volume collapse**: The tariff does achieve trade deficit reduction, but primarily through reduced overall trade engagement rather than export promotion. By 2055, the trade deficit in tariff scenarios is approximately half that of non-tariff scenarios, but cumulative trade volume is 25-30% lower.

#### 6.2.3 Long-Term Consequences (2055-2075)

The final decades of the simulation reveal the permanent effects of the tariff regime:

- **Persistent output gap**: By 2075, real GDP in tariff scenarios remains approximately 8-9% below non-tariff scenarios, representing a permanent loss of economic efficiency worth over $1.5 trillion annually in real output.

- **Lingering structural vulnerability**: Surprisingly, the AI Decision Makers tariff scenario shows another episode of extreme volatility in 2067-2068, with growth spiking to 9.9% before contracting by 7.0%, suggesting that even after decades of adaptation, the tariff-oriented economy remains structurally vulnerable to shocks.

- **Fiscal legacy**: The debt-to-GDP ratio differential reaches 15-18 percentage points by 2075, with tariff scenarios at 203-205% versus 185-188% in non-tariff scenarios, creating a substantial fiscal burden for future generations.

- **Policy performance differential**: Throughout the entire 50-year simulation, the AI Decision Makers approach consistently achieves better outcomes than traditional policy frameworks under both tariff and non-tariff scenarios, with lower volatility and better management of trade-offs between competing objectives.

### 6.3 Policy Insights and Implications

The comprehensive analysis across both historical comparison and the tariff shock scenario yields several important policy insights:

#### 6.3.1 Optimal Policy Characteristics

The AI-learned policies consistently demonstrate several key characteristics:

1. **Forward-looking adjustment**: The AI approach begins policy adjustments 1-2 quarters before economic turning points become fully evident, highlighting the importance of anticipatory rather than reactive policymaking.

2. **Responsiveness over permanence**: The optimal policies feature faster, more decisive adjustments that are calibrated to evolving conditions rather than maintaining rigid policy stances regardless of economic context.

3. **Coordinated policy mix**: The AI approach consistently demonstrates effective coordination across policy instruments, with fiscal and monetary tools working in complementary rather than offsetting directions.

4. **Counter-cyclical fiscal stance**: During economic expansions, the AI policies build fiscal capacity through moderately higher tax rates and disciplined spending growth, creating space for aggressive support during downturns.

5. **Non-linear response functions**: The learned policies show non-linear responses to economic conditions, with disproportionately strong reactions to extreme scenarios while maintaining stability during normal fluctuations.

#### 6.3.2 Distributional Considerations

The historical simulation reveals important insights regarding distributional effects:

1. **Labor market prioritization**: The AI approach consistently achieves better unemployment outcomes during recessions without sacrificing long-term inflation control, suggesting a more favorable Phillips curve trade-off.

2. **Transfer payment targeting**: The optimal policies implement larger but more temporary transfer payment increases during downturns, providing stronger support when most needed while avoiding permanent dependency.

3. **Balanced growth model**: The AI approach facilitates a more balanced growth model with less reliance on consumption and greater emphasis on investment and exports, potentially supporting more sustainable and broadly shared prosperity.

#### 6.3.3 Trade Policy Implications

The tariff shock analysis provides several important insights regarding trade policy:

1. **Efficiency-protection trade-off**: While tariffs do reduce trade deficits and protect certain domestic industries, they impose substantial economy-wide efficiency costs that compound over time, ultimately reducing rather than enhancing prosperity.

2. **Fiscal illusion**: Despite generating direct revenue, high tariffs ultimately worsen fiscal positions by shrinking the broader tax base, increasing transfer dependency, and raising interest costs, creating a fiscal burden that outweighs the tariff revenue.

3. **Structural vulnerability**: Even after decades of adaptation, tariff-oriented economies show persistent vulnerability to shocks, with episodes of extreme volatility emerging even 40+ years after tariff implementation.

4. **Differential adaptation capacity**: The AI Decision Makers approach demonstrates superior management of the tariff shock compared to traditional policy frameworks, achieving better stability and welfare outcomes despite the challenging environment.

## 7. Conclusion

This paper has introduced a novel framework integrating the FRB/US macroeconomic model with Proximal Policy Optimization reinforcement learning enhanced by an active relocation mechanism. Our approach combines the structural economic relationships embedded in FRB/US with the adaptive learning capabilities of reinforcement learning, creating a powerful tool for fiscal policy design.

Our results demonstrate that this integrated approach consistently outperforms both historical policy decisions and traditional model-based frameworks across diverse economic conditions. The AI-learned policies exhibit greater countercyclical responsiveness, more effective coordination across policy instruments, and better management of trade-offs between competing objectives.

The historical comparison spanning five decades (1975-2024) reveals that our approach would have achieved superior macroeconomic outcomes, with less severe recession impacts, stronger recoveries, and better long-term fiscal sustainability. The tariff shock analysis extending to 2075 demonstrates the framework's capacity for long-term policy optimization even under extreme scenarios, while highlighting the substantial and persistent economic costs of protectionist trade policies.

Several important policy insights emerge from our analysis:

1. **Timing matters**: Optimal fiscal policy begins responding before economic turning points become fully evident, highlighting the value of forward-looking adjustment.

2. **Flexibility enhances effectiveness**: Responsive, state-contingent policies outperform rigid rules, adapting quickly to evolving conditions while maintaining consistent long-term objectives.

3. **Policy coordination is crucial**: Effective fiscal policy operates in conjunction with, rather than opposition to, monetary policy, enhancing overall macroeconomic stability.

4. **Building fiscal capacity is essential**: Disciplined fiscal management during expansions creates space for aggressive support during downturns, improving both short-term stability and long-term sustainability.

### 7.1 Limitations and Future Work

While our framework represents a significant advancement in fiscal policy design, several limitations suggest directions for future research:

1. **Model dependency**: Our results are conditioned on the structural relationships embedded in FRB/US. Future work should explore robustness across alternative macroeconomic models and explicit model uncertainty.

2. **Political economy constraints**: Real-world policy implementation faces political constraints not captured in our framework. Incorporating political feasibility considerations could enhance practical applicability.

3. **Distributional analysis**: More detailed examination of distributional effects across household types and economic sectors would provide richer insights into welfare implications.

4. **Multi-agent dynamics**: Extending the framework to model strategic interactions between domestic fiscal authorities, monetary policymakers, and international counterparts could yield important insights into global policy coordination 

5. **Adaptive expectations**: While our framework incorporates model-consistent expectations, further exploration of how private sector learning and adaptation affect policy outcomes would enhance realism.

6. **Structural change**: Longer-term simulations should account for potential structural changes in the economy, including demographic shifts, technological transformation, and climate-related transitions.

### 7.2 Policy Implications

Our findings have several important implications for practical fiscal policy design:

1. **Institutional framework**: Establishing institutional mechanisms that facilitate more responsive fiscal policy adjustment while maintaining long-term objectives could significantly enhance macroeconomic stability. This might include automatic stabilizers that adjust based on forward-looking indicators rather than lagging measures.

2. **Forecast integration**: Integrating high-quality economic forecasts more systematically into fiscal policy decisions could help achieve the forward-looking adjustment patterns that characterize optimal policies in our framework.

3. **Policy coordination**: Creating stronger coordination mechanisms between fiscal and monetary authorities while preserving appropriate independence could enhance overall policy effectiveness, as demonstrated by the complementary policy patterns in our AI approach.

4. **Fiscal capacity**: Prioritizing fiscal consolidation during economic expansions creates essential policy space for aggressive countercyclical support during downturns, suggesting the importance of rebuilding fiscal buffers as economic conditions permit.

5. **Trade policy considerations**: The substantial and persistent costs of high tariffs highlighted in our analysis suggest policymakers should carefully weigh protectionist measures against their long-term effects on economic efficiency, fiscal sustainability, and structural vulnerability.

### 7.3 Concluding Remarks

The integration of structural macroeconomic models with reinforcement learning represents a powerful new approach for policy design, combining economic theory with data-driven adaptation. Our framework demonstrates that this combination can discover policy strategies that effectively navigate complex trade-offs while respecting economic constraints.

The consistent outperformance of our AI-enhanced approach across both historical comparison and hypothetical stress tests suggests significant potential for improving real-world policy outcomes. By systematically exploring the vast policy space and learning from interaction with a high-fidelity economic model, our framework identifies non-obvious strategies that balance multiple competing objectives more effectively than conventional approaches.

As policymakers face increasingly complex challenges—from pandemic recovery to climate transition to technological disruption—tools that enhance policy design while respecting economic fundamentals become increasingly valuable. Our integrated approach offers such a tool, providing a flexible framework for discovering robust policy strategies across diverse economic conditions.

While our current implementation focuses on fiscal policy optimization, the approach could be extended to other policy domains including monetary policy, financial regulation, and structural reform. The combination of structural economic understanding with adaptive learning capabilities represents a promising direction for economic policy research, bridging traditional macroeconomic modeling with frontier developments in artificial intelligence.
