# Fiscal Policy Strategy: Integrating FRB/US with Reinforcement Learning and Active Relocation

## Abstract

Fiscal policy to optimize competing macroeconomic objectives presents significant challenges for economic policymaking. Although the Federal Reserve’s FRB/US model provides sophisticated forecast, its reliance on predefined scenarios limits exploration of the full policy space. This research introduces the RL-FRB/US model which integrates the FRB/US model and Proximal Policy Optimization (PPO) reinforcement learning (RL) model with an active enhancement of relocation mechanism for fiscal policy optimization. The five-decade comprehensive analysis (1975-2024) based on the RL-FRB/US model demonstrates significant performance improvements over baseline FRB/US simulations. By 2024Q2, the RL-FRB/US model achieved higher real GDP (23,407 vs. 23,218), lower unemployment (3.23\% vs. 3.96\%), and more effective inflation management (PCPI 317.9 vs. 312.3). During recessions, the model consistently delivered superior counter-cyclical responses, with unemployment peaks significantly reduced during major downturns—during the 1982 recession, peak unemployment reached only 9.9\% compared to 10.9\% in traditional simulations. While the RL-FRB/US model showed similar federal budget deficits by 2024 (-\$1,767B vs. -\$1,758B), it achieved substantially lower debt-to-GDP ratios (26,535 trillion \$ vs. 30,186 trillion \$) through more strategic debt management during expansionary periods. The outputs indicates that a combination of reinforcement learning and macroeconomic modeling introduces more reliable outputs than the traditional model, that provides policymakers with powerful decision-support instruments to balance inflation control, targeted unemployment rate and fiscal sustainability.

## 1. Introduction

Macroeconomic policymaking requires balancing multiple competing objectives while navigating complex, uncertain environments. Fiscal policy design is particularly challenging as policymakers must simultaneously consider growth, employment, inflation, debt sustainability, and distributional impacts. Traditional approaches to fiscal policy analysis rely on large-scale macroeconomic models with manually specified policy rules and scenarios, potentially missing optimal policy configurations in the vast multidimensional policy space.

The Federal Reserve Board's FRB/US model represents the state-of-the-art in structural macroeconomic modeling, with a rich history of guiding U.S. monetary and fiscal policy decisions. It features sophisticated descriptions of household, firm, and financial market behavior, capturing both short-run dynamics and long-run equilibrium properties. However, the conventional application of such models typically involves analyzing a small set of predefined policy alternatives, rather than systematically exploring the full range of possible policy responses across diverse economic conditions.

Fiscal policy aims are to promote economic growth, stabilize other macro indicators such as employment, inflation, fund rate. However, it is challenging for policymakers to optimize simultaneously targets under uncertain environment. Practically, traditional methods to construct fiscal policy heavily rely on large-scale macroeconomic models, which potentially miss optimal policy configurations in the vast multidimensional policy space. 
In the US, the Federal Reserve Board has been applying the FRB/US model, the state-of-the-art in structural macroeconomic modeling, which plays as a guideline for FED and Treasury to build and implement monetary and fiscal policies. Based on sophisticated descriptions of household, firm, and financial market behavior, both agencies can forecast short-run dynamics and long-run equilibrium. However, this model relies only on static assumptions, which does not cover the dynamic conditions and external forces. Therefore, the model seems inefficient estimation in some extreme conditions such economic shocks or external shock. 
Currently, reinforcement learning (RL) has been applied in economic forecast offering better outcomes in complex environments. RL methods are designed for sequential decision-making under uncertainty, learning through repeated interaction with an environment to discover policies that maximize cumulative rewards. Among these methods, Proximal Policy Optimization (PPO) has emerged as particularly effective for high-dimensional control problems due to its sample efficiency and stability.
This research introduces RL-FRB/US model - a novel framework that integrates the FRB/US macroeconomic model with PPO reinforcement learning enhanced by an active relocation mechanism. As a result, the method outperforms over traditional ones: 

    
- Systematic policy space exploration: Rather than evaluating a handful of predefined policy alternatives, the model systematically explores high-dimensional fiscal policy configurations, identifying strategies that might be overlooked in conventional analysis.

- Adaptive policy responses: The learned policies adapt to changing economic conditions, providing state-contingent guidance rather than fixed rules.

- Balance of multiple objectives: The reward function can incorporate multiple policy objectives, allowing for explicit tradeoffs between competing goals such as growth, inflation, and fiscal sustainability.

- Enhanced learning efficiency: The active relocation mechanism enables the agent to strategically navigate the economic state space during training, focusing on informative or challenging scenarios to accelerate convergence.



Therefore, the outputs of this combination are to create a powerful instrument for fiscal policy design that can discover non-obvious policy strategies while respecting economic constraints.

The research applies RL-FRB/US model to investigate 25-year (2000-2024) and five-decade (1975-2024) historical data to compare its and the traditional models’ outputs,  interestingly, the new model operates much better than the traditional one.

### 3.1 Basic Concepts in Reinforcement Learning

Reinforcement learning is a model in which an agent learns to make sequential decisions by interacting with an environment. The agent’s goal is to maximize its cumulative reward over time. In the context of fiscal policy, the key components of an RL model include:

- **State (s)**: The observable information about the environment, in this case, a vector of macroeconomic indicators from FRB/US model.
- **Action (a)**: The decision made by the agent at time \(t\). In the model, actions correspond to adjustments in fiscal policy instruments (e.g., changes in tax rates, government expenditures).
- **Reward (r)**: A scalar signal that quantifies the desirability of the outcome following an action. The reward is typically designed to penalize deviations from target macroeconomic objectives (such as an inflation target or a sustainable debt-to-GDP ratio).
- **Policy (π)**: A function, parameterized by \(\theta\), that maps states to actions. Modern implementations typically use deep neural networks to represent this policy.
- **Value Function (V)**: The expected cumulative reward when starting from state \(s\) and following policy \(\pi\) thereafter.

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

$$a_t = [\Delta ge_t, \Delta tx_{p,t}, \Delta tx_{c,t}]$$

where:
- $\Delta ge_t$ denotes the change in government expenditures, bounded by $[-0.25, 0.25]$ billion dollars
- $\Delta tx_{p,t}$ captures the adjustment to personal tax rates, bounded by $[-0.1, 0.1]$
- $\Delta tx_{c,t}$ represents the change in corporate tax rates, bounded by $[-0.1, 0.1]$

The PPO agent outputs a stochastic policy $\pi_\theta(a|s)$ modeled as a multivariate Gaussian distribution with state-dependent means $\mu_\theta(s)$ and standard deviations $\sigma_\theta(s)$:

$$\pi_\theta(a|s) = \mathcal{N}(a|\mu_\theta(s), \sigma_\theta(s)^2)$$

This stochastic policy allows for exploration during training while gradually converging toward deterministic policy choices as uncertainty decreases.

### 4.4 Reward Function Formulation
The RL-FRB/US model's reward function deliberately targets real GDP growth as its singular optimization objective based on extensive experimental validation. This targeted approach leverages the fundamental interconnectedness of macroeconomic indicators, where robust GDP growth serves as a central driver that naturally optimizes auxiliary metrics. Experimental results consistently demonstrate that when the model successfully maximizes sustainable growth trajectories, it simultaneously achieves superior outcomes for unemployment, inflation, and fiscal sustainability without explicitly targeting these variables. This phenomenon occurs because policies that genuinely enhance productivity and economic efficiency—rather than artificially stimulating specific sectors—create positive spillover effects throughout the economic system. The model's success with this simplified reward structure confirms Cochrane's assertion that addressing growth fundamentals provides the most direct path to comprehensive economic improvement, allowing policymakers to focus on removing structural impediments rather than managing complex tradeoffs between competing objectives.

The reward function component 

$$\delta_{g}(t) = \text{clamp}((g_t - g^*)/10, -1, 1)$$

measures the deviation between realized GDP growth ($g_t$) and the target growth rate ($g^*$). This formulation quantifies how successfully monetary and fiscal policy decisions approach the desired economic growth trajectory. When actual growth exceeds the target, the function yields a positive value, signaling optimal policy performance; conversely, growth below the target produces a negative value, indicating suboptimal outcomes. The magnitude of this deviation directly influences the overall reward signal, creating a clear incentive structure that guides the reinforcement learning algorithm toward policies that consistently generate growth rates approaching or exceeding the established benchmark ($g^*$). This straightforward yet powerful mechanism enables the model to learn complex policy responses that maximize sustainable economic expansion.

The RL-FRB/US model adopts a 4\% economic growth target ($g^*$ = 4\%) for its reward function based on compelling evidence that this level represents an optimal balance between historical achievement and aspirational policy goals. As Cochrane convincingly argues, while the U.S. economy historically grew at 3.5\% annually from 1950-2000, achieving 4\% growth would yield "even greater benefits" beyond this post-WWII norm. The model's reward structure acknowledges the profound impact such growth rates have on living standards—Cochrane illustrates that a 1.5 percentage point difference in growth compounds dramatically over time, potentially doubling per capita income within a generation. Furthermore, robust growth serves as a fiscal stabilizer; as Cochrane notes, if GDP grows at 3.5\% instead of the CBO's projected 2.2\%, "GDP in 2040 would be 38\% higher, tax revenues would be 38\% higher, and a lot of the problem would go away on its own." This insight directly informed the model design, as the 4\% target creates a reward function that incentivizes policies capable of addressing multiple economic challenges simultaneously through growth enhancement rather than tradeoff management. By establishing this ambitious yet historically-grounded benchmark, the RL-FRB/US model explicitly values policies that remove structural impediments to productivity and innovation—the true engines of sustainable long-term prosperity.

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

The RL-FRB/US model utilized the official FRB/US model as the simulation environment, integrating the Proximal Policy Optimization (PPO) algorithm as described by Schulman et al. (2017). The research's  implementation draws from the historical macroeconomic time series (HISTDATA.TXT) covering 1975-2024 with quarterly observations of over hundreds of economic indicators mentioned in Section \ref{sec:litreview}. The Federal Reserve dataset provides comprehensive information on output, inflation, employment, interest rates, fiscal indicators, and international trade, serving both as initial conditions for simulations and for calculating model residuals that capture stochastic components of economic relationships.

The agent's architecture - using Spinning Up implementation - consists of the dual neural network structure: the policy network (actor) and the value function network (critic). Both networks share the same architecture—three fully connected hidden layers of 128 units with layer normalization and ReLU activations—but are independently parameterized. The actor network outputs both the mean and standard deviation for each action dimension, enabling stochastic policy sampling, while the critic network estimates the state value function to guide the learning process.

Training proceeds through sequential quarterly economic decisions, with the agent interacting with the FRB/US model by observing economic states, taking fiscal policy actions, and receiving feedback. For the training of the PPO/Active Relocation model, the historical data from (1975-1999) were used as the training data, and the validation data will be from 2000-2024. 


For both the historical setting analysis (2000 - 2024) for validation purposes, and the longue duree setting (1975 - 2024), the research will illustrate three scenarios:


- Historical Data: Actual economic outcomes with historical policy decisions
- FRB/US Simulation: Counterfactual simulation using traditional FRB/US with baseline policy rules
- RL-FRB/US: The integrated method with RL-optimized fiscal policy


The training was conducted using the following hyperparameters:
- Learning rate: 0.0003
- Discount factor ($\gamma$): 0.99
- GAE parameter ($\lambda$): 0.95
- Clipping parameter ($\epsilon$): 0.2
- Value function coefficient: 1
- Entropy coefficient: 0.01
All simulations were implemented in PyTorch and conducted with a fixed random seed for reproducibility.

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
