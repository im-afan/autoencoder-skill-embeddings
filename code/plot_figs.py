import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd

if __name__ == "__main__":
    import tensorboard as tb
    from tbparse import SummaryReader
    highlevel_logdirs = ["./tensorboard/AntObstacleHighLevel/AntObstacleHighLevel-v0/PPO_4/events.out.tfevents.1698818131.Yipings-MacBook-Pro.local.10176.0",
                        "./tensorboard/AntObstacleHighLevel/AntObstacleHighLevel-v0/PPO_5/events.out.tfevents.1698890743.Yipings-MacBook-Pro.local.20924.0"]
    lowlevel_logdirs = ["./tensorboard/AntObstacleLowLevel/AntObstacleLowLevel-v0/PPO_2/events.out.tfevents.1698694302.46a9b078cff8.1042.0",
                        "./tensorboard/AntObstacleLowLevel/AntObstacleLowLevel-v0/PPO_3/events.out.tfevents.1698874336.Yipings-MacBook-Pro.local.15376.0"]

    data_lowlevel, data_highlevel = {}, {}
    for dir in highlevel_logdirs:
        reader = SummaryReader(dir)
        df = reader.scalars
        print(df)
        for index, row in df.iterrows():
            if(row["tag"] != "rollout/ep_rew_mean"):
                continue
            if(row["step"] not in data_highlevel):
                data_highlevel[row["step"]] = row["value"]
            else:
                #data_highlevel[row["step"]] += row["value"]
                #data_highlevel[row["step"]] /= 2
                data_highlevel[row["step"]] = max(data_highlevel[row["step"]], row["value"])
    for dir in lowlevel_logdirs:
        reader = SummaryReader(dir)
        df = reader.scalars
        print(df)
        for index, row in df.iterrows():
            if(row["tag"] != "rollout/ep_rew_mean"):
                continue
            if(row["step"] not in data_lowlevel):
                data_lowlevel[row["step"]] = row["value"]
            else:
                data_lowlevel[row["step"]] = max(data_lowlevel[row["step"]], row["value"])

    
    x, y = [], []
    for timestep in data_highlevel:
        x.append(timestep), y.append(data_highlevel[timestep])

    x1, y1 = [], []
    for timestep in data_lowlevel:
        x1.append(timestep), y1.append(data_lowlevel[timestep])

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(7, 5))

    print(x, y)
    df = pd.DataFrame({"timestep": x+x1, "reward": y+y1, "run": [0 if i < len(x) else 1 for i in range(len(x)+len(x1))]})

    sns.relplot(data=df, x="timestep", y="reward", hue="run", kind="line")
    #sns.relplot(data=df1, x="timestep", y="reward", kind="line")
    plt.show()


