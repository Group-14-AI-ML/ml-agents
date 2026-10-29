1. # Project 2.1 General Idea

   1. ## Vocabulary & Core Concepts

   2. ### What Is Machine Learning?

Machine Learning is a branch of artificial intelligence where computers discover patterns and rules directly from data, rather than having a programmer write explicit instructions for every possible scenario: in traditional programming, a developer writes manual conditional logic (like if-else rules) to determine how a system responds, while in Machine Learning, the system receives examples of inputs paired with real outcomes, and it automatically calculates mathematical relationships that allow it to make predictions or decisions on new, unseen examples.

1. ### What Is Deep Learning?

Deep Learning is a specialized subfield of Machine Learning based on multi-layered artificial neural networks. While standard Machine Learning models work effectively with flat, tabular data using straightforward mathematical equations, they struggle with high-dimensional, raw data streams. Deep Learning solves this by passing numbers through stacked layers of interconnected virtual nodes, loosely modeled after biological neurons. Each layer extracts progressively higher-level concepts from raw numbers—translating sensor readings, image pixels, or spatial coordinates into abstract representations that inform final actions. We'll treat Deep Learning as a black box.

2. ### What Is a Model?

A model is the resulting mathematical function or representation produced by this learning process. Before learning occurs, an untrained model is simply an empty mathematical architecture filled with arbitrary, randomized values called weights. Once trained, the model represents the concrete decision-making program: a system that accepts an input (such as sensor numbers) and calculates an output (such as choosing which action to execute or predicting a final number).

3. ### How Training Works

Training is an iterative optimization process that refines the internal weights of a model through systematic trial and adjustment:

1. **The Forward Pass:** The model takes an input from the current environment state, processes it through its internal weights, and produces an initial decision or prediction.

2. **Measuring Error or Performance:** The system compares what the model did with what was desired, calculating a numerical loss (the difference between prediction and reality) or observing a scalar reward signal.

3. **The Backward Pass (Backpropagation):** The learning algorithm calculates how much each individual weight in the network contributed to the error or success.

4. **Weight Updates:** The algorithm makes a slight adjustment to the network weights, nudging them in the direction that reduces error or increases future rewards.

5. **Repetition:** By repeating this loop across thousands or millions of interactions, the internal weights stabilize into a policy capable of performing the task reliably.

### How We Use It in Our Project

In our project, we do not train an AI model from scratch to solve an everyday problem; instead, we combine these principles across two distinct levels to save time and computing power.

Our primary machine learning challenge sits one level higher. Because running these 3D deep learning trials consumes substantial computer power and clock time, running endless sessions blindly is impractical. To solve this, we record the conditions and outcomes of our simulation runs into a structured dataset. We then train our own supervised machine learning model on that data. This secondary model acts as a surrogate predictor: by analyzing pre-run setup configurations, it estimates runtime duration and task success beforehand, bypassing the need to launch an expensive simulation every single time.

To see where this telemetry data originates, we must look directly inside the specific Unity testing ground chosen for this project: the Push Block environment.

4. ### Environment, Observations, Actions, and Rewards

**The agent** is the moving character, while **The environment** is Push Block, including the movable block, goal, boundary walls, and physical rules: but how the agent can sense and observe the environment? Through **ray-cast**, an invisible beam sent out in a specific direction to detect what object it touches (wall, block, or goal) and how far away it is: the agent uses **14 ray-casts** that together produce 70 continuous observation numbers that will lead to an **action**, a discrete choices: move forward, backward, left, right, rotate clockwise, rotate counterclockwise, or do nothing. Based on the taken action, the agent will get a **rewards** or a **penalty** (+1 for hitting the goal, \-0.0025 per step) during the training phase (more of this later).

5. ### Parameters vs. Hyperparameters vs. Environment Settings

**Parameters** or **model parameters** are internal weights inside the neural network that the training algorithm optimizes automatically, while **Hyperparameters:** are some settings we choose manually before launching the training run, here are some examples:

* *Learning Rate:* The step size taken when updating neural weights. Too large causes chaotic learning and failure; too small makes training take far too long.

* *Batch Size:* The number of observed simulation experiences bundled together before running a learning update.

* *Network Size:* Number of layers and hidden units in the neural policy.

* *Step Budget:* The total number of steps allocated to a training run. We also have **environment Parameters** that are physical properties of the game world itself (such as block\_scale, dynamic\_friction, static\_friction, and block\_drag)

## Research Questions

* **RQ1 (Success Rate Prediction):**  
  *How accurately can we predict the final evaluation success rate of an ML-Agents Push Block training run from its pre-training configuration, including block scale, selected training hyperparameters, and compute budget?*  
  *Target:*

* **RQ2 (Runtime & Stability Analysis):**  
  *How do variations in batch size and learning rate affect the total training time and policy stability of the PPO agent in PushBlock?*  
  *Target:*

## What We Are Going to Do Across the Three Phases

2. ### Phase 1: Setup and Planning

Each teammate installs Unity, configures an isolated Python 3.10 virtual environment, and clones the course ML-Agents fork. We set up our shared public GitHub repository and test that everyone can commit and pull updates. Instead of running long simulations, we verify our logging workflow by doing a short test run in Push Block, making sure our scripts record basic hardware information and elapsed timestamps into a local file without crashing. Once verified, we finalize our written Project Plan.

3. ### Phase 2: Running Games and Gathering Data

Our team runs multiple Push Block sessions under different knob combinations (varying batch sizes, learning speeds, and block scales). Because individual training runs take between 20 minutes and over an hour, we run sessions in the background on our own computers. We log the pre-training settings used, machine hardware details, elapsed runtimes, and final success rates. We then share our records with the class-wide repository and download data submitted by other student groups. Finally, we merge these files into a single structured spreadsheet where every row represents one complete training run.

4. ### Phase 3: Training the Predictor and Answering the Research Questions

We move away from running Unity simulations and focus on our spreadsheet. Using Python, we keep about 80% of the rows to teach a supervised machine learning predictor and reserve the remaining 20% to test its accuracy on unseen runs. The predictor learns how the setup knobs correlate with training time and task success, finishing its calculations in seconds. We compare its guesses against the true values from our test set, checking that it performs significantly better than a simple average guess. Lastly, we inspect which setup knobs had the greatest influence on runtime and stability to answer our research questions in the final report.