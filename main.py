from GradientBandit import GradientBandit
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,10)
import numpy as np
import time

#EXPERIMENT
N_BANDITS = 2000
N_STEPS = 1000
ALPHA = [0.05,0.1,0.4]


def run_experiment(k=10,n_bandits=500,n_steps=500,alpha=[0.05,0.1,0.4]):
    optimal_action_perc_dict = {}
    q_star_a = np.random.normal(4,1,k)
    print(f"True Q values: {q_star_a}")
    for bandit_n in range(len(alpha)):
        bandit_wrb_T = GradientBandit(q_star_a,k,n_bandits,n_steps,alpha[bandit_n],with_reward_baseline=True)
        bandit_wrb_F = GradientBandit(q_star_a,k,n_bandits,n_steps,alpha[bandit_n],with_reward_baseline=False)

        key = "Gradient bandit with alpha="+str(alpha[bandit_n])+" with Reward Baseline"
        optimal_action_perc_dict[key] = bandit_wrb_T.run_bandit()
        key = "Gradient bandit with alpha="+str(alpha[bandit_n])+" without Reward Baseline"
        optimal_action_perc_dict[key] = bandit_wrb_F.run_bandit()
        
        
    return optimal_action_perc_dict

start_time = time.perf_counter()
optimal_action_perc_dict = run_experiment(n_bandits=N_BANDITS,n_steps=N_STEPS,alpha=ALPHA)
end_time = time.perf_counter()

print(f"Execution time: {end_time - start_time:.6f} seconds")

#Plot
for experiment_result_key in optimal_action_perc_dict.keys():
    print(experiment_result_key)
    data = optimal_action_perc_dict[experiment_result_key]
    plt.plot(data,label=experiment_result_key);
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    plt.legend();

#plt.show()

plt.savefig("GradientBandit_Run.svg", format='svg', dpi=300)
plt.close()  # Close the plot to free memory

print('Plot saved to disk')


