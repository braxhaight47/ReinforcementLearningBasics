import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
# Imports for environment and neural network set up^
# --------------------------------------------------------------------------------

# Author: Braxton Haight, Final model: Thursday, March 31st

# --------------------------------------------------------------------------------
# Environment setup
# Create environment name (Use one from the storage)
environment_name = 'CartPole-v0'
# Create the environment using the prebuilt space
env = gym.make(environment_name)
# How many times AI runs for the practice session to show results
episodes = 5

# --------------------------------------------------------------------------------
# Testing
# Begin learning/ Testing environment
for episode in range(1, episodes + 1):
    # Set environment to base
    state = env.reset()
    # Set done to False so when done is True loop ends
    done = False
    # Score counter, start at 0
    score = 0
# Make loop for rendering the environment and having the AI train for episodes
    while not done:
        # Render the environment
        env.render()
        # Create randomized action
        action = env.action_space.sample()
        # Set the results of the action (these vars are the outcomes)
        n_state, reward, done, info = env.step(action)
        # Add reward to score to store the total okf rewards
        score += reward
    # Print out the run number and the score for that run
    print('Episode: {} Score: {}'.format(episode, score))
# Close the environment
# env.close

# --------------------------------------------------------------------------------
# Training logs
# Make these directories
log_path = os.path.join('Training', 'Logs')

# Recreated environment in gym space
env = gym.make(environment_name)

# Wraps the env in a dummy vectorized env so that testing can occur
env = DummyVecEnv([lambda: env])
# Defines policy using neural network (PPO), and the setting (MlpPolicy)
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
# Number of times the model learns (Min. 20,000)
model.learn(total_timesteps=20000)

# --------------------------------------------------------------------------------
# Saving the model
# Save model data to path
PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')
# Save
model.save(PPO_Path)

# Delete model
del model

# Reload model
model = PPO.load(PPO_Path, env=env)

# Testing and evaluation
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

# --------------------------------------------------------------------------------
# Run model on observations
# Num of runs
episodes = 5

# Loop for runs
for episode in range(1, episodes + 1):
    # Set observations to blank environment
    obs = env.reset()

    # Set done to False
    done = False
    # Sets score value to zero
    score = 0
# Make loop for rendering the environment and having the AI train for episodes
    while not done:
        # Render the environment
        env.render()
        # Create predicted action based on observations
        action, _ = model.predict(obs)
        # Set the results of the action (these vars are the outcomes)
        obs, reward, done, info = env.step(action)
        # Add reward to score to store the total okf rewards
        score += reward
    # Print out the run number and the score for that run
    print('Episode:{} Score:{}'.format(episode, score))
# Closes the environment
env.close()

# --------------------------------------------------------------------------------
# Callbacks
# Reset observations
obs = env.reset()
# Set the results of the action (these vars are the outcomes)
action, _ = model.predict(obs)
# Set environment action to random sample
env.action_space.sample()
# Preform action
env.step(action)
# Create training log path
training_log_path = os.path.join(log_path, 'PPO_0')

# --------------------------------------------------------------------------------
# Optimal model creation
# Creates save path for best model
save_path = os.path.join('Training', 'Saved Models')
# Stops calling back when an average reward of 200 is reached
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
# Will call back every 10000 steps and save the best model to the save model paths
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=save_path,
                             verbose=1)
# Recreates model as before
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
# New learn will be with call back

# Creates best model
model.learn(total_timesteps=20000, callback=eval_callback)
# Final result is now the best model

# End of ideal model build
# --------------------------------------------------------------------------------
# Start of trying different algorithms and neural networks

# --------------------------------------------------------------------------------
# Changing policies
# Switching architects (neural network)
net_arch = [dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])]
# Set the model to running the new policy
model = PPO('MlpPolicy',  env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch': net_arch})
# Learn method using the callback method from above, but using the new neural network
model.learn(total_timesteps=20000, callback=eval_callback)
# --------------------------------------------------------------------------------
# End of testing/creating reinforcement trained AI using stable_baseline3 environment space.
