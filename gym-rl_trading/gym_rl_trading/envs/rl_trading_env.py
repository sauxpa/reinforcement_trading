# !svn export https://github.com/sauxpa/stochastic/trunk/ito_diffusions

import abc
import gym
from gym import spaces
import numpy as np
from collections import deque
from ito_diffusions import FBM
from statsmodels.graphics.tsaplots import acf

ONE_PCT = 1e-4

class RLTradingEnv(gym.Env):
    """A synthetic stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 n_lag=500,
                 init_balance=0,
                 max_steps=1000,
                 horizon=2.0,
                 reward_mode='pnl',
                 ):
        super(RLTradingEnv, self).__init__()

        # at each step, observe n_lag prices
        self.n_lag = n_lag
        # initial cash amount
        self.init_balance = init_balance
        # maximum step in the environment
        self.max_steps = max_steps
        # time horizon (in years)
        self.horizon = horizon

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Discrete(3)

        # Prices contains the returns for the last n_lag prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_lag, 6), dtype=np.float64)

        # Either 'pnl' or 'sharpe'
        self.reward_mode = reward_mode

    @abc.abstractmethod
    def new_return(self):
        pass

    @property
    @abc.abstractmethod
    def description(self):
        pass

    @property
    def dt(self):
        return self.current_step / self.max_steps * self.horizon

    def _next_observation(self):
        # Get the stock data points for the last self.n_lag days

        self.obs.popleft()
        new_return = self.new_return()
        self.obs.append(new_return)
        self.last_price = self.price
        self.price += new_return
        return self.obs

    def _take_action(self, action):
        current_price = self.price
        self.last_balance = self.balance
        self.last_shares_held = self.shares_held
        amount = 1

        if action == 0:
            # Buy amount % of balance in shares
            shares_bought = amount
            self.shares_held += shares_bought
            cost = shares_bought * current_price
            self.balance -= cost

        elif action == 1:
            # Sell amount % of shares held
            shares_sold = amount
            self.shares_held -= shares_sold
            gain = shares_sold * current_price
            self.balance += gain

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1

        obs = self._next_observation()

        self.pnl = self.shares_held * self.price \
        - self.last_shares_held * self.last_price \
        + self.balance \
        - self.last_balance

        self.pnl_history.append(self.pnl)

        if self.reward_mode == 'pnl' or np.std(self.pnl_history) <= 1e-3:
            reward = self.pnl
        elif self.reward_mode == 'sharpe':
            reward = self.pnl / (np.std(self.pnl_history))
        else:
            raise Exception('{} not an allowed reward mode.'.format(self.reward_mode))

        done = self.current_step >= self.max_steps

        return obs, reward, done, {}

    def reset_returns(self):
        self.obs = deque()
        self.current_step = 0
        for _ in range(self.n_lag):
            self.current_step += 1
            self.obs.append(self.new_return())

        self.init_price = 0.9 + 0.2 * np.random.rand()
        self.price = np.sum(self.obs) + self.init_price

    def reset(self):
        """ Reset the state of the environment to an initial state.
        """
        self.balance = self.init_balance
        self.shares_held = 0

        self.pnl = 0
        self.pnl_history = []

        self.reset_returns()

        return self._next_observation()


class RLTradingEnvBM_NoTrend(RLTradingEnv):
    def __init__(self):
        super().__init__()
        self.drift = 0 * ONE_PCT
        self.vol = 20 * ONE_PCT

    @property
    def description(self):
        return 'Brownian prices, no trend, no profitable strategy.'

    def new_return(self):
        return self.drift * self.dt \
        + self.vol * np.random.randn() * np.sqrt(self.dt)


class RLTradingEnvBM_Trend(RLTradingEnv):
    def __init__(self):
        super().__init__()
        self.drift = 5 * ONE_PCT
        self.vol = 20 * ONE_PCT

    @property
    def description(self):
        return 'Brownian prices, +5% trend, profitable long strategy.'

    def new_return(self):
        return self.drift * self.dt \
        + self.vol * np.random.randn() * np.sqrt(self.dt)


class RLTradingEnvBM_Cyclical(RLTradingEnv):
    def __init__(self):
        super().__init__()
        self.drift = 10  * ONE_PCT
        self.vol = 20 * ONE_PCT
        self.n_cycles = 10

    @property
    def description(self):
        return 'Brownian prices carried by a cyclical trend. Profitable long-short strategy based on the position in the cycle.'

    def new_return(self):
        return self.drift * np.sin(2*np.pi/self.horizon*self.dt*self.n_cycles) * self.dt \
        + self.vol * np.random.randn() * np.sqrt(self.dt)


class RLTradingEnvFBM(RLTradingEnv):
    def __init__(self):
        super().__init__(n_lag=5)
        self.drift = 0 * ONE_PCT
        self.vol = 5

    @property
    @abc.abstractmethod
    def H(self):
        """Hurst Index.
        """
        pass

    def reset_returns(self):
        self.init_price = 90 + 20 * np.random.rand()

        X = FBM(x0=self.init_price,
                T=self.horizon,
                scheme_steps=self.max_steps,
                drift=self.drift,
                vol=self.vol,
                H=self.H,
                )

        df = X.simulate()
        df_ret = (df-df.shift(1)).iloc[1:]

        self.prices = np.array(df['spot'])
        self.returns = np.array(df_ret['spot'])

        self.obs = deque()
        self.current_step = 0
        for _ in range(self.n_lag):
            self.current_step += 1
            self.obs.append(self.new_return())

        self.price = np.sum(self.obs) + self.init_price

    def new_return(self):
        return self.returns[self.current_step-1]


class RLTradingEnvFBMBaseline(RLTradingEnvFBM):
    """Deterministic baseline:
    1) Evaluate autocorrelation on the last observed returns,
    2) If returns are empirically autocorrelated:
        -> buy if rallye, sell if sell-off
        Otherwise do the opposite.
    """
    def __init__(self):
        super().__init__()

    def _take_action(self, action):
        current_price = self.price
        self.last_balance = self.balance
        self.last_shares_held = self.shares_held

        amount = 1

        autocorr = acf(self.returns[:self.current_step], fft=True)[1]
        if autocorr > 0:
            # momentum
            action = 0 if self.obs[-1] > 0 else 1
        else:
            # mean-reversion
            action = 1 if self.obs[-1] > 0 else 0

        if action == 0:
            # Buy amount % of balance in shares
            shares_bought = amount
            self.shares_held += shares_bought
            cost = shares_bought * current_price
            self.balance -= cost

        elif action == 1:
            # Sell amount % of shares held
            shares_sold = amount
            self.shares_held -= shares_sold
            gain = shares_sold * current_price
            self.balance += gain


class RLTradingEnvFBM_07(RLTradingEnvFBM):
    def __init__(self):
        super().__init__()

    @property
    def description(self):
        return 'Fractional Brownian prices, with local trend-following patterns.'

    @property
    def H(self):
        """Hurst Index.
        """
        return 0.7


class RLTradingEnvFBM_03(RLTradingEnvFBM):
    def __init__(self):
        super().__init__()

    @property
    def description(self):
        return 'Fractional Brownian prices, with local mean-reversion patterns.'

    @property
    def H(self):
        """Hurst Index.
        """
        return 0.3


class RLTradingEnvFBMBaseline_07(RLTradingEnvFBMBaseline):
    def __init__(self):
        super().__init__()

    @property
    def description(self):
        return 'Deterministic baseline with local trend-following patterns.'

    @property
    def H(self):
        """Hurst Index.
        """
        return 0.7


class RLTradingEnvFBMBaseline_03(RLTradingEnvFBMBaseline):
    def __init__(self):
        super().__init__()

    @property
    def description(self):
        return 'Deterministic baseline with local mean-reversion patterns.'

    @property
    def H(self):
        """Hurst Index.
        """
        return 0.3


class RLTradingEnvFBMAutoCorr(RLTradingEnvFBM):
    def __init__(self):
        super().__init__()
        self.n_autocorr = 3
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.n_lag+self.n_autocorr, 6),
            dtype=np.float64,
            )

    def _next_observation(self):
        self.obs_ret.popleft()
        new_return = self.new_return()

        self.obs_ret.append(new_return)
        self.last_price = self.price
        self.price += new_return

        obs = np.concatenate(
            [
                np.array(self.obs_ret),
                acf(self.obs_ret, fft=True)[1:self.n_autocorr+1],
            ]
        )

        return obs

    def reset_returns(self):
        self.init_price = 90 + 20 * np.random.rand()

        X = FBM(x0=self.init_price,
                T=self.horizon,
                scheme_steps=self.max_steps,
                drift=self.drift,
                vol=self.vol,
                H=self.H,
                )

        df = X.simulate()
        df_ret = (df-df.shift(1)).iloc[1:]

        self.prices = np.array(df['spot'])
        self.returns = np.array(df_ret['spot'])

        self.obs_ret = deque()
        self.current_step = 0
        for _ in range(self.n_lag):
            self.current_step += 1
            self.obs_ret.append(self.new_return())

        self.price = np.sum(self.obs_ret) + self.init_price


class RLTradingEnvFBMAutoCorr_07(RLTradingEnvFBMAutoCorr):
    def __init__(self):
        super().__init__()

    @property
    def description(self):
        return 'Fractional Brownian prices, with local trend-following patterns. Observations are augmented vith the empirical autocorrelogram.'

    @property
    def H(self):
        """Hurst Index.
        """
        return 0.7


class RLTradingEnvFBMAutoCorr_03(RLTradingEnvFBMAutoCorr):
    def __init__(self):
        super().__init__()

    @property
    def description(self):
        return 'Fractional Brownian prices, with local mean-reversion patterns. Observations are augmented vith the empirical autocorrelogram.'

    @property
    def H(self):
        """Hurst Index.
        """
        return 0.3
