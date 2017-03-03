# Scalable Multiagent Coordination with Distributed Online Open Loop Planning

We propose distributed online open loop planning (DOOLP), a general framework for online multiagent coordination and decision making under uncertainty. DOOLP is based on online heuristic search in the space defined by a generative model of the domain dynamics, which is exploited by agents to simulate and evaluate the consequences of their potential choices.
We also propose distributed online Thompson sampling (DOTS) as an effective instantiation of the DOOLP framework. DOTS models sequences of agent choices by concatenating a number of multiarmed bandits for each agent and uses Thompson sampling for dealing with action value uncertainty. The Bayesian approach underlying Thompson sampling allows to effectively model and estimate uncertainty about (a) own action values and (b) other agents' behavior. This approach yields a principled and statistically sound solution to the exploration-exploitation dilemma when exploring large search spaces with limited resources.
We implemented DOTS in a smart factory case study with positive empirical results. We observed effective, robust and scalable planning and coordination capabilities even when only searching a fraction of the potential search space. 

Paper on arXiv: https://arxiv.org/pdf/1702.07544.pdf
