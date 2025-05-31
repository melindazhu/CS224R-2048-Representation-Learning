# CS224R-2048-Representation-Learning
CS224R Custom Project - using deep RL / representation learning to efficiently play 2048

## Setup
Copy the cs224r_project_main.ipynb notebook into Colab and run the cells.

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