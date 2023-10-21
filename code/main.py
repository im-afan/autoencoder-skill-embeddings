import numpy as np
import subprocess
import project_config

subprocess.run("ls")

for seed in range(0, 3):
    np.seed(seed)
    folder = "trials/" + str(seed)
    with open("cur_path.txt", "w") as f:
        f.write(folder)
    subprocess.run(["mkdir", folder])
    subprocess.run(["mkdir", folder + "/autoencoders"])
    subprocess.run(["mkdir", folder + "/logged_states/"])
    subprocess.run(["mkdir", folder + "/logged_states/anttargetpos"])
    project_config.DECODER_PATH = folder + "/autoencoders/decoder.pth"
    subprocess.run(["python3", "train_lowlevel.py", "AntTargetPos", folder])
    subprocess.run(["python3", "test_lowlevel.py", "AntTargetPos", folder])
    subprocess.run(["python3", "train_autoencoder.py", folder + "/autoencoders/", folder + "/logged_states/anttargetpos/"])
    subprocess.run(["python3", "train_highlevel.py", "AntTargetPos", "--log-folder", folder + "/agents"])
    subprocess.run(["python3", "train_highlevel.py", "AntObstacle", "--log-folder", folder + "/agents"])
    subprocess.run(["python3", "train_lowlevel.py", "AntObstacle", folder])
