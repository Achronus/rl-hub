# Reinforcement Learning Basics

RL tells you how to make the best decisions sequentially, within a context, to maximise a real-life measure of success.

It comprises of four main components:

- **Agent**: a decision-making entity that learns through trial and error.
- **Action**: the decision made by the agent.
- **Environment**: a situation the agent is constrained to. Can be a simulation, real-life environment, or a combination of both. These accept actions and respond with a new set of observations.
- **Reward**: encodes the challenge, acting as a feedback mechanism to tell the agent which actions lead to success (or failure). Used to reinforcement behaviour and is typically numeric.

These components represent a *Markov Decision Process (MDP)* which are used to frame problems, even non-engineering ones.

## Advantages

RL's primary advantage is that it optimizes for long-term, multi-step rewards.

Another, is that it's very easy to incorporate metrics used by the business. For example, advertising solutions are typically optimized to promote the best click-through rates for individual ads. This is suboptimal. Viewers often see multiple ads, but the goal isn't to get a click, it's something bigger like user retention, a sign-up, or a purchase. The combination of advertisements shown, in what order, with what content, can all be optimzied automatically by RL utilizing an easy-to-use goal that matches the business' needs.

## Reward Engineering

You don't always need to use the complete set of components for an MDP. If you don't have a natural reward signal, like a robot reaching a goal, you can engineer an artificial one.

You can also create a simulation of an environment or quantize and truncate actions. These are all compromises - a simulation can never replace real-life experience.

## Common Problems

RL agents suffer because they often optimise for the wrong thing. Typically, it's better to keep the reward as simple as possible. Problems often have a natural reward.

## Main Elements

- **Policy**: the decision and planning process of the agent (its brain) - decides the actions the agent takes during each step, denoted by $\pi$.
- **Reward Function**: determines the amount of reward an agent receives after completing a action or series of actions. Reward is often given externally, but some interal reward systems exist.
- **Value Function**: determines the value of a state over the long term - fundamental to RL.
- **Environment Models**: a model represents a fully observable environment, such as representing all possible game states in tic-tac-toe. Most RL algorithms use the concept of a partially observable state, due to the number of states in an environment (e.g., more states than number of atoms in the universe).

## Markov Decision Process (MDP)

A discrete-time stochastic control process.

- **Discrete**: time moves forward in finite intervals:

    $$
    t \in {1, 2, 3, \cdots}
    $$

- **Stochastic**: future states depend only partially on the actions taken
- **Control process**: based on decision making to reach the target state

Mathematical components: $(S, A, R, P)$,:

- $S$: set of possible states of the task
- $A$: set of actions that can be taken in each of the states
- $R$ set of rewards for each state-action $(s, a)$ pair
- $P$: the probabilities of passing from each state to another when taking each possible action

MDPs have no memory. The next state depends only on the current state, not on the previous ones. This process is known as a *Markov process* and fulfills the *Markov Property*.

$$
P[S_{t+1} | S_t = s_t] = P[S_{t+1} | S_t = s_t, S_{t-1} = s_{t-1}, \cdots, S_0 = s_0]
$$

MDPs are an extension of *Markov Chains*, MDPs introduce *actions* and *rewards*.

### Types of MDPs

- **Finite**: the number of $S$, $A$ and $R$ are finite.
- **Infinite**: one or more of the number of $S$, $A$, and $R$ is infinite.
- **Episodic**: MDPs last for a finite set of rounds with a terminal state. Episodes are a *trajectory* from the initial state of the task to a terminal state.

$$
\tau = S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, \cdots, R_T, S_T
$$

- **Continuous**: MDPs have no terminal state and continue forever.

## Components

- **Trajectory**: elements that are generated when the agent moves from one state to another.

$$
\tau = S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, S_3
$$

- **Reward**: the immediate result an action ($A_t$) produces. The goal of any RL task is represented by the *rewards* ($R_t$) of the environment. Sometimes a short-term reward can worsen long-term results, so we want to maximise the *long-term* sum of rewards.
- **Return**: the sum of rewards that the agent obtains up to a certain point in time $T$ until the task is completed. The long-term result for the actions we take during an episode. We want to maximise the *episode's return*.

$$
G_t = R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T
$$

### Discount Factor

Denoted by $\gamma \in [0, 1]$. Measures how far into the future the agent has to look when planning its actions.

Adds an incentive to the agent to learn the shortest route to the goal. Applied to returns to help the agent learn during longer timesteps.

$$
G_0 = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4 + \cdots + \gamma^{T-t-1} R_T
$$

If $\gamma = 0$, all future rewards will be 0. Encourages agents to act greedily, only taking the action that provides the highest immediate reward.

If $\gamma = 1$, rewards are not discounted. Agent has more patience to formulate a long-term strategy.

We want to maximise the long-term sum of *discounted* rewards.

### Policy

A function that decides what action to take in a particuluar state.

$$
\pi: S \mapsto A
$$

- $\pi(a | s)$: the probability of taking action $a$ in state $s$
- $\pi(s)$: action $a$ taken in state $s$

There are two types:

- **Stochastic**: a probability distribution over a range of actions.

$$
\pi(s) = [p(a_1), p(a_2), \cdots, p(a_n)]
$$

$$
\pi(s) = [0.3, 0.2, 0.5]
$$

- **Deterministic**: a mapping from state to action. These always choose the same action in a given state.

$$
\pi(s) \to a
$$

$$
\pi(s) = a_1
$$

Goal: we want to maximise the sum of discounted rewards by finding an optimal policy $\pi_*$.

### Value Functions

- **State-value**: the value of a state is defined as the return we expect to obtain starting from that state $s$ and interacting with the environment following policy $\pi$ until the terminal state.

$$
v_\pi(s) = \mathbb{E}[G_t | S_t = s]
$$

$$
v_\pi(s) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1} R_T | S_t = s]
$$

- **Action-value**: the q-value of an action in a state is the return that we expect to obtain if we start in state $s$ and take action $s$ then interact with the environment following policy $\pi$ until the terminal state.

$$
q_\pi(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a]
$$

$$
q_\pi(s, a) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1} R_T | S_t = s, A_t = a]
$$

### Bellman Equation

For the *state-value* function:

$$
\begin{align}
v_\pi(s) &= \mathbb{E}[G_t | S_t = s] \\
&= \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1} R_T | S_t = s] \\
&= \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
&= \sum_{a} \pi(a | s) \sum_{s', r} \mathbb{P}(s,r | s,a) [r + \gamma v_{\pi}(s')]
\end{align}
$$

We can decompose the *state-value* function to get the return from starting from the next moment in time, discounted by gamma. We can then simplify this (3) and decompose it further.

In (4), we get the probability of taking each action when following a policy, multipled by the return we expect to get from taking that action. We can then express the probability of reaching each possible successor state ($s'$) multipled by the reward obtained after reaching that state + the discounted value of the next state $\gamma v_{\pi}(s')$.

We can repeat this same process for the *action-value* function:

$$
\begin{align}
q_\pi(s,a) &= \mathbb{E}[G_t | S_t = s, A_t = a] \\
&= \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1} R_T | S_t = s, A_t = a] \\
&= \mathbb{E} [R_{t+1} + \gamma G_{t+1} | S_t = A_t = a] \\
&= \sum_{s', r} \mathbb{P}(s',r | s,a) \left[ r + \gamma \sum_{a'} \pi(a' | s') q_{\pi} (s', a') \right]
\end{align}
$$

In (4), we can rewrite the expected return as the probability of reaching each successor state $s'$ given a chosen action $a$, multipled by the first reward obtained upon reaching that next state $s'$ + the discounted sum of the q-values used for each action in the next state, weighted by the probability of choosing that action by a given policy.

Note: both equations have recursive relationships.

### Solving an MDP

Goal: sovling a control task consists of maximising the expected return.

The optimal value of a state is the expected return following the optimal policy.

In *state-value* functions, it's to maximise the value of every state.

$$
v_*(s) = \mathbb{E}_{\pi_*} [G_t | S_t = s]
$$

In *action-value* functions, it's to maximise the value of every q-value.

$$
q_*(s,a) \mathbb{E_{\pi_*}} [G_t | S_t = s, A_t = a]
$$

To maximise their returns, we have to find the optimal policy $\pi_*$, which is the policy that takes the optimal actions that maximises $v(s)$ or $q(s,a)$.

The optimal policy is defined as the policy that takes actions in each state that lead to the maximium expected return.

With *state-value* functions, we take into account the states where the action leads ($\sum_{s', r} p(s',r | s,a)$) and the return that we expect to obtain in the next state ($[r + \gamma v_*(s)]$).

$$
\pi_*(s) = \argmax_{a} \sum_{s', r} p(s',r | s,a) [r + \gamma v_*(s)]
$$

With *action-value* functions, we simply select the action whose q-value is highest.

$$
\pi_*(s) = \argmax_{a} q_*(s,a)
$$

However, there is a problem - to find the optimal policy $\pi_*$ we have to know the optimal values. To find the optimal values $v_*$ or $q_*$, we need to know the optimal policy.

With both depending on each other, how do we solve this? We can use *bellman optimality equations*:

$$
v_*(s) = \max_{a} \sum_{s', r} \mathbb{P}(s',r | s,a) [r + \gamma v_*(s')]
$$

$$
q_*(s,a) = \sum_{s', r} \mathbb{P}(s',r | s,a) \left[ r + \gamma \max_{a'} q_* (s', a') \right]
$$
