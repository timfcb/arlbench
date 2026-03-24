import matplotlib.pyplot as plt

def save_timestep(timestep, global_step):
    file1 = open("image.txt", "a")
    file1.write(f"Global step: {global_step}, Last Obs: {timestep.last_obs}, Action: {timestep.action}, Obs: {timestep.obs}, Reward: {timestep.reward}, Done: {timestep.done}\n")
    file1.close()


def save_timestep_eval(timestep):

    last_obs, action, obs, reward, done, info = timestep
    file1 = open("image.txt", "a")
    file1.write(f"Last Obs: {last_obs}, Action: {action}, Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info} \n")
    file1.close()


def plot_imgs(obs):
    plt.imshow(obs)
    plt.axis('off')
    plt.show()
    plt.savefig("plot.pdf", format="pdf")  


def save_obs(observation):
    file1 = open("obs.txt", "a")
    file1.write(f"Observation: {observation}\n")
    file1.close()
