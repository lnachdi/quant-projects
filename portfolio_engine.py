import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats

def get_data(tickers,start_date,end_date):
	"""
	Downloads the closing prices and computes the log returns of the assets.

	Parameters:
	tickers: list of ticker strings
	start_date: string "YYYY-MM-DD"
	end_date: string "YYYY-MM-DD"

	Returns:
		prices: DataFrame of daily closing prices
		log_returns: Dataframe of daily logarithmic returns
	"""
	data=yf.download(tickers,start=start_date, end=end_date, auto_adjust=True)
	prices=data['Close'].dropna()
	log_returns=np.log(prices/prices.shift(1)).dropna()
	return prices, log_returns

def annualized_stats(log_returns):
	"""
	Computes annualized expected returns and covariance matrix

	Parameters:
		log_returns: DataFrame of daily log returns

	Returns:
		mean_returns: Series of annaul expected returns
		cov_matrix: DataFrame of annual cov matrix
	"""
	mean_returns=log_returns.mean() *252
	cov_matrix = log_returns.cov() * 252
	return mean_returns, cov_matrix

def portfolio_preformance(weights,mean_returns,cov_matrix,risk_free_rate=0.047):
	"""
	Computes annualized return, volatility, and Sharpe ratio for a portfolio

	Parameters:
		weights: array of portfolio weights
		mean_returns: Series of annaul expected returns
		cov_matrix: DataFrame of annual cov matrix
		risk_free_rate: annual risk free rate (default set at 0.047)

	Returns:
		port_return: float of annualized expected return
		port_vol: float of annualized volatility
		sharpe: float of Sharpe ratio
	"""
	port_return= weights @ mean_returns	
	port_vol= np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))
	sharpe = (port_return - risk_free_rate)/port_vol
	return port_return, port_vol, sharpe

def minimum_variance_portfolio(mean_returns,cov_matrix):
	"""
	Finds the portfolio weights that minimize volatility

	Returns:
		weights: array of optimal weights
		port_return: float
		port_vol: float
	"""

	n=len(mean_returns)

	def objective(w):
		return np.sqrt(np.dot(w.T,np.dot(cov_matrix,w)))

	constraints = {'type': 'eq',"fun": lambda w: np.sum(w)-1}
	bounds = tuple((0,1) for _ in range(n))
	w0 = np.array([1/n]*n)

	result = minimize(objective,w0,method="SLSQP",bounds=bounds,constraints=constraints)

	weights = result.x
	port_return = np.dot(weights,mean_returns)
	port_vol= objective(weights)
	return weights, port_return,port_vol

def maximum_sharpe(mean_returns,cov_matrix,risk_free_rate=0.047):
	"""
	Finds portfolio that maximizes Sharpe ratio

	Returns:
		weights: array of optimal weights
		port_return: float
		port_vol: float
		sharpe: float
	"""
	n=len(mean_returns)

	def negative_sharpe(w):
		ret=np.dot(w,mean_returns)
		vol=np.sqrt(np.dot(w.T,np.dot(cov_matrix,w)))
		return -(ret-risk_free_rate)/vol

	constraints = {"type":'eq', "fun": lambda w: np.sum(w)-1}
	bounds=tuple((0,1) for _ in range (n))
	w0 = np.array([1/n]*n)

	result = minimize(negative_sharpe,w0,method="SLSQP",bounds=bounds,constraints=constraints)

	weights=result.x
	port_return,port_vol,sharpe= portfolio_performance(weights,mean_returns,cov_matrix,risk_free_rate)
	return weights,port_return,port_vol,sharpe

def efficient_frontier(mean_returns,cov_matrix, n_points=100):
	"""
	Computes the efficient frontier and optimizes for a range of target returns

	Returns:
		frontier_returns: array of traget returns
		frontier_vols: array of minimized volatilities
	"""

	n=len(mean_returns)

	min_w,min_ret,min_vol=minimum_variance_portfolio(mean_returns,cov_matrix)
	max_ret=mean_returns.max()
	target_rets=np.linspace(min_ret,max_ret,n_points):
	frontier_vols=[]

	for target in target_rets:
		constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w)-1},{'type': 'eq','fun': lambda w, t=target: np.dot(w,mean_returns)-t}]
		bounds = tuple((0,1) for _ in range(n))
		w0= np.array([1/n]*n)

		result=minimize(lambda w: np.sqrt(np.dot(w.T,np.dot(cov_matrix,w))), w0, method="SLSQP",bounds=bounds,constraints=constraints)
		frontier_vols.append(result.fun)
	return target_rets,np.array(frontier_vols)


def monte_carlo_sim(optimal_weights,log_returns,initial_value=10000,n_simulations=1000,n_days=252,risk_free_rate=0.047):
	"""
	Will run Monte Carlo forward simulation and compute VaR and CVaR

	Returns:
		simulation_results: array (n_days,n_simulations)
		var_historical: float at 95% hist VaR
		var_parametric: float at 95% parametric VaR
		cvar_historical: float at 95% historical CVaR
		historical_port_returns: Series of portfolio returns

	"""

	np.random.seed(42)
	daily_cov=log_returns.cov()
	



