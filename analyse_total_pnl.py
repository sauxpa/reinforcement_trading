import numpy as np
import pandas as pd
import argparse, sys
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# Get arguments
parser=argparse.ArgumentParser()
parser.add_argument('--filename', '-f', type=str)
parser.add_argument('--plot', '-p', default=False, type=bool)
parser.add_argument('--title', '-t', default='', type=str)

# Set arguments
args = parser.parse_args()
filename = args.filename
plot = args.plot
title = args.title

df = pd.read_csv(filename)
df = df.set_index('Step')
df = df.drop('Wall time', axis='columns')
df['Cumulative PnL'] = df['Value'].cumsum()


total_pnl = df['Cumulative PnL'].iloc[-1]
nsteps = df.index[-1]


print('Total PnL of {:,.0f} over {} steps'.format(total_pnl, nsteps))
if plot:
    fig, ax = plt.subplots(figsize=(11, 4), nrows=1, ncols=1)
    df['Cumulative PnL'].plot(ax=ax)
    if title == '':
        ax.set_title('Total PnL of {:,.0f} over {} steps'.format(total_pnl, nsteps))
    else:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
