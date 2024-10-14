# RL Use Cases Notes

## When To Use

RL works best when decisions are sequential and actions lead to exploration in the environment.

RL actively searches for the optimal model. You don't have to generate a random sample and fit it offline. Rapid, online learning works wonders when it is important to maximise performance as quickly as possible.

RL is best suited to applications that need sequential, complex decisions and have a long-term goal (for a single decision). It excels in environments with direct feedback.

## Examples

1. Robotics is a classic RL application. The goal of a robot is to learn how to perform unknown tasks. You shouldn't tell the robot how to succeed, it's often too difficult to explain (e.g., how do you tell a robot to build a house?) or you can bias their approach based on your own experience.

    We are not robots so how do we know the best way to move like one? If we instead allow the robot to explore, it can iterate toward an optimal solution.

2. In profit-based A/B tests, like deciding what marketing copy to use, you don't want to waste time collecting a random sample if one underperforms. RL does this automatically.

3. It's even possible to replace teams of data scientists that tweak performance out of ML solutions.

4. Other robotics applications include: improving movement and manufacturing, playing ball-in-a-cup, flipping pancakes, and autonomous vehicles - [1](https://www.mdpi.com/2218-6581/2/3/122), [2](https://arxiv.org/abs/2001.03864).

5. It can be used to improve cloud computing, such as latency, power efficiency/usage, datacenter cooling, CPU cooling, and network routing.

6. The financial industry - to make trades, to perform portfolio allocations, or optimizing pricing in real-time.

7. The amount of energy used by buildings (through heating, water, light, and so on) and electricity grids to deal with situations where demand is complex; homes are both producers and consumers - [1](https://arxiv.org/abs/1903.05196), [2](https://dl.acm.org/doi/10.1145/3485128).

8. It can be used to improve traffic light control and active lane management - [1](https://ieeexplore.ieee.org/document/5658157), [2](https://ieeexplore.ieee.org/document/6338837).

9. Smart cities could also benefit - [1](https://ieeexplore.ieee.org/document/7945258).

10. Healthcare, especially areas of dosing and treatment schedules [1](https://www.media.mit.edu/publications/reinforcement-learning-for-designing-novel-clinical-trials-for-treating-cancer-patients/), [2](https://ieeexplore.ieee.org/document/8031178). Also, for better prosthetics and prosthetic controllers [3](https://www.semanticscholar.org/paper/Transfer-Learning-for-Prosthetics-Using-Imitation-Mohammedalamen-Khamies/71e6e18c52cb72fc6aff54f1e81b3c34b0c0b015).

11. Education systems and e-learning for highly individualized RL-driven curriculums.

Gaming, technology, transport, finance, science and nature, industry, manufacturing, and civil services all have cited RL applications. It's possibilities are endless.
