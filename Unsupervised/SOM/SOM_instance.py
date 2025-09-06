from SOM_algo import SOM
import pandas as pd

df = pd.read_csv("data/WineQT.csv")

som = SOM(df)
som.train(
    learning_rate=0.5,
    neighborhood_radius=3,
    lr_decay_rate=0.01,
    nr_decay=0.01,
    percent_df=0.8,
    T=50,
    grid_size=10
)

som.plot_mapping(df, labels=pd.Series(df["quality"]))
