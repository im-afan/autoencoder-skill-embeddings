import numpy as np
import subprocess

for seed in range(0, 4):
    np.random.seed(seed)
    subprocess.run("ls")
    #subprocess.run(["python3", "-V"])
    subprocess.run("python3 train_lowlevel.py AntTargetPos --log-folder trials/agents".split(" "))
    subprocess.run("python3 test_lowlevel.py AntTargetPos --folder trials/agents".split(" "))
    subprocess.run("python3 train_autoencoder.py trials/autoencoders/ trials/logged_states/anttargetpos/".split(" "))
    subprocess.run("python3 train_highlevel.py AntTargetPos --log-folder trials/agents".split(" "))
    subprocess.run("python3 train_highlevel.py AntObstacle --log-folder trials/agents".split(" "))