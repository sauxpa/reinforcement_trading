from gym.envs.registration import register

# Brownian prices, no trend, no profitable strategy.
register(
    id='rl_trading-v0',
    entry_point='gym_rl_trading.envs:RLTradingEnvBM_NoTrend',
)

# Brownian prices, +5% trend, profitable long strategy.
register(
    id='rl_trading-v1',
    entry_point='gym_rl_trading.envs:RLTradingEnvBM_Trend',
)

# Brownian prices carried by a cyclical trend. Profitable long-short
# strategy based on the position in the cycle.
register(
id='rl_trading-v2',
entry_point='gym_rl_trading.envs:RLTradingEnvBM_Cyclical',
)

# Fractional Brownian prices, with local trend-following patterns.
register(
    id='rl_trading-v3',
    entry_point='gym_rl_trading.envs:RLTradingEnvFBM_07',
)

# Fractional Brownian prices, with local mean-reversion patterns.
register(
    id='rl_trading-v4',
    entry_point='gym_rl_trading.envs:RLTradingEnvFBM_03',
)

# Deterministic baseline with local trend-following patterns.
register(
    id='rl_trading-v5',
    entry_point='gym_rl_trading.envs:RLTradingEnvFBMBaseline_07',
)

# Deterministic baseline with local mean-reversion patterns.
register(
    id='rl_trading-v6',
    entry_point='gym_rl_trading.envs:RLTradingEnvFBMBaseline_03',
)

# Fractional Brownian prices, with local trend-following patterns.
# Observations are augmented vith the empirical autocorrelogram.
register(
    id='rl_trading-v7',
    entry_point='gym_rl_trading.envs:RLTradingEnvFBMAutoCorr_07',
)

# Fractional Brownian prices, with local mean-reversion patterns.
# Observations are augmented vith the empirical autocorrelogram.
register(
    id='rl_trading-v8',
    entry_point='gym_rl_trading.envs:RLTradingEnvFBMAutoCorr_03',
)
