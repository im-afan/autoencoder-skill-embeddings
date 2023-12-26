import numpy as np
import subprocess
import project_config

subprocess.run("ls")

def trial(seed, in_embed_state=True, latent_embed_state=True):
    #np.random.seed(seed)
    folder = "trials/" + str(seed)


    subpath = str(int(in_embed_state)) + str(int(latent_embed_state))
    with open("cur_path.txt", "w") as f:
        f.write(folder + "\n")
        f.write(subpath)

    subprocess.run(["mkdir", folder])
    subprocess.run(["mkdir", folder + "/autoencoders"])
    subprocess.run(["mkdir", folder + "/autoencoders/" + subpath])
    subprocess.run(["mkdir", folder + "/logged_states/"])
    subprocess.run(["mkdir", folder + "/logged_states/reachwithgripper"])
    #project_config.DECODER_PATH = folder + "/autoencoders/decoder.pth"
    #subprocess.run(["python3", "train_lowlevel.py", "AntTargetPos", folder])
    subprocess.run(["python3", "train_autoencoder.py", folder + "/autoencoders/", folder + "/logged_states/reachwithgripper/", subpath])
    #subprocess.run(["python3", "train_highlevel.py", "AntTargetPos", "--log-folder", folder + "/agents"])
    subprocess.run(["python3", "train_highlevel.py", "PickUp", folder + "/agents"])


for seed in range(3, 5):
    """ 
    np.random.seed(seed)
    folder = "trials/" + str(seed)
    with open("cur_path.txt", "w") as f:
        f.write(folder)
    subprocess.run(["mkdir", folder])
    subprocess.run(["mkdir", folder + "/autoencoders"])
    subprocess.run(["mkdir", folder + "/logged_states/"])
    subprocess.run(["mkdir", folder + "/logged_states/anttargetpos"])
    #project_config.DECODER_PATH = folder + "/autoencoders/decoder.pth"
    #subprocess.run(["python3", "train_lowlevel.py", "AntTargetPos", folder])
    subprocess.run(["python3", "test_lowlevel.py", "AntTargetPos", folder])
    subprocess.run(["python3", "train_autoencoder.py", folder + "/autoencoders/", folder + "/logged_states/anttargetpos/"])
    #subprocess.run(["python3", "train_highlevel.py", "AntTargetPos", "--log-folder", folder + "/agents"])
    subprocess.run(["python3", "train_highlevel.py", "AntObstacle", "--log-folder", folder + "/agents"])
    subprocess.run(["python3", "train_lowlevel.py", "AntObstacle", folder])"""
    folder = "trials/" + str(seed)
    subprocess.run(["mkdir", folder])
    subprocess.run(["python3", "train_lowlevel.py", "ReachWithGripper", folder])
    subprocess.run(["python3", "test_lowlevel.py", "ReachWithGripper", folder])
    trial(seed, True, True)
    trial(seed, True, False)
    trial(seed, False, True)
    trial(seed, False, False)
    #subprocess.run(["python3", "train_lowlevel.py", "PickUp", folder])
