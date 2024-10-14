# Taxonomy Of RL

## Model-Free or Model-Based

The first major decision you need to make when working on an RL problem is: *do you need an accurate model of the environment?*

*Model-based* algorithms use definitive knowledge of the environment they operate in to improve learning.

For example, board games often limit the moves that you can make, and you can use this knowledge to:

- Constrain the algorithm so that it doesn't provide invalid actions
- Improve performance by projecting forward in time (e.g., if I move here and if my opponent moves there, I can win)

A limited set of moves = limited number of strategies the algorithm has to search through. Model-based approaches learn efficiently because they don't waste time searching improper paths.

However, most environments are complex and cannot model an environment due to it's state size, more on this later.

*Model-free* algorithms can, in theory, apply to any problem. They learn strategies through interaction, absorbing any environmental rules in the process.

You can even get hybrid algorithms that learn models of the environment at the same time as learning optimal strategies. Some can even leverage the potential, but unknown actions of other agents/players, such as counteracting another agent's strategies.

## Agent Strategies

The goal of an agent is to learn a strategy (policy $\pi$) that maximises a reward.

How and when an algorithm updates the policy is the defining factor between the majority of model-free RL algorithms.

There are two key factors to consider, first the difference between their updates. Second, the type of strategy that defines the action selection.

### Online vs Offline Agents

Agent behaviour - how an agent performs a policy update.

*Online* agents improve their policy using only the data they have just observed and then immediately throw it away. They don't store or reuse old data.

All RL agents need to update their policy when they encounter a new experience to some extent, but most state-of-the-art algorithms agree that retaining and reusing past experience is useful.

*Offline* agents learn from offline datasets or previous experiences. This can be beneficial because sometimes it is difficult or expensive to interact with the real-world.

However, RL tends to be most useful when agents learn online, so most algorithms aim for a hybrid approach.

### On-Policy vs Off-Policy

Policies - the strategy for how agents select their actions.

*On-policy* agents learn to predict the reward of being in states after choosing actions according to the current strategy. They learn a new policy using the one they've already seen as a starting point. These tend to learn quicker than their counter-part because they can instantly exploit new strategies.

*Off-policy* agents learn to predict the reward after choosing *any* action. They use the same policy as a starting point, but is allowed to explore other actions with a little bit of randomness. Off-policy is often favoured because of this reason, it encourages/improves exploration and tends to work better on tasks with delayed rewards.

Modern algorithms favour hybrid approaches to these too.

## Discrete vs Continuous Actions

Actions can take many forms and are often split into two types: *discrete* or *continuous*.

*Discrete* actions are finite options, such as moving [up, down, left, right].

*Continuous* actions are slightly more rigorous and often involve numeric ranges, such as [-1, 1] for controlling a steering wheel.

## Optimisation Methods

*Value-based* algorithms involve trying as many actions as possible, recording the results and then use that information to guide the agent by following a strategy that leds to the best one.

*Policy-based* algorithms maintain a model and tweak the parameters of them to achieve the actions that produce the best result (e.g., Neural Networks).

*Imitation-based* algorithms optimise an agent's performance by mimicing the actions of an expert. These can work well when you are trying to incorporate human guidance.

## Policy Evaluation and Improvement

A simplified interpretation for improving an agent's policy goes as follows:

Imagine an agent follows a policy to make decisions which generates new data that describes the state of its environment. From this new data, the agent attempts to predict the reward from the current state of the environment by evaluating the current policy.

Next, the agent uses the prediction to decide what to do next. The agent could suggest to move to a state with a higher predicted reward, or explore the environment more. Repeating this behaviour, it tries to change it's strategy to improve the policy.

![Algorithm process](/rl/assets/imgs/simple-process.png)

The vast majority of algorithms follow this same pattern.
