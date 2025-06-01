# CS224R-2048-Representation-Learning
CS224R Custom Project - using deep RL / representation learning to efficiently play 2048

## Setup
Copy the cs224r_project_main.ipynb notebook into Colab and run the cells.

## Experiments implemented (6/1/25)
Before running any of these experiments

1. **Upgrade the Q Network to use a CNN** to reshape the input board into a spatially-aware grid from a flattened vector. [Commit](https://github.com/melindazhu/CS224R-2048-Representation-Learning/commit/036b87a42b5435a2bf813fa85ca9dce9fb02377b)

2. **Add auxiliary task to predict legal actions and reward shaping to penalize illegal actions**: help the agent learn spatial representations that can distinguish between legal and illegal actions, reducing the number of wasted steps. Essentially, the task predicts the legality of each action from the current state, improving the ability to avoid bad moves even when Q-values are ambiguous or noisy. [Commit](https://github.com/melindazhu/CS224R-2048-Representation-Learning/commit/b606606a5deea708500f970538dcedee0996bef5)

3. **Add auxiliary task to predict the max tile**: encourage the agent to learn state representations that are predictive of long-term success; the vanilla DQN only optimizes for short-term rewards. The auxiliary task teaches the network to predict the maximum tile that will be achieved by the end of the episode from the current state. As a result, the agent learns to represent states in a way that captures more long-term potential. [Commit](https://github.com/melindazhu/CS224R-2048-Representation-Learning/commit/96ed500303aa39a8374a1cbc6f2f969bc482cbb1)

4. **Legal masking**: get rid of illegal action options completely. Unknown whether this helps, but might be worth a try. Might be hindering the ability to explore. [Commit](https://github.com/melindazhu/CS224R-2048-Representation-Learning/commit/374013dbe6da3d62108438adc31538838e20be4a)

5. **Add gradient clipping**: added gradient clipping to help with exploding gradients (a consequence of this is that the loss shoots up). After gradient clipping was added, the loss improved a lot more and was no longer continuously increasing throughout training. [Commit](https://github.com/melindazhu/CS224R-2048-Representation-Learning/commit/96ed500303aa39a8374a1cbc6f2f969bc482cbb1)

## For experimentation:
Download the specific commit for the improvement added.

Instructions for pulling the changes from specific commits:

To access a specific commit, go to the commit history in GitHub.
```
# First, checkout a new branch locally

$ git checkout -b my_exp_branch

# Then, cherry pick the specific commit

$ git cherry-pick commit_hash
```

If you're adding a lot of separate evaluation code that might affect the training loop (or if you're just testing stuff), I'd recommend just creating a separate branch and then creating a pull request when you're ready to commit changes like so:

```
$ git checkout -b new_branch_name

# Make your changes...

$ git add --all

$ git commit -m "my_commit_description"

$ git push -u origin my_branch_name
```