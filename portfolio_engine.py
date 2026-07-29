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

def portfolio_performance(weights,mean_returns,cov_matrix,risk_free_rate=0.047):
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
	target_rets=np.linspace(min_ret,max_ret,n_points)
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
	simulation_results=np.zeros((n_days,n_simulations))

	for sim in range(n_simulations):
		daily_returns=np.random.multivariate_normal(mean=log_returns.mean(),cov=daily_cov,size=n_days)
		portfolio_daily=daily_returns @ optimal_weights
		price_path= initial_value * np.exp(np.cumsum(portfolio_daily))
		simulation_results[:,sim] = price_path

	historical_port_returns = log_returns @ optimal_weights
	var_hist = np.percentile(historical_port_returns,5)
	cvar_hist = historical_port_returns[historical_port_returns <= var_hist].mean()

	mu=historical_port_returns.mean()
	sigma=historical_port_returns.std()
	var_parametric = stats.norm.ppf(0.047,mu,sigma)

	return (simulation_results, var_hist, cvar_hist,var_parametric,historical_port_returns)

def plot_analysis(tickers, prices,log_returns,mean_returns,cov_matrix,min_weights,max_weights,frontier_returns,frontier_vols,
	simulation_results,var_hist,cvar_hist,historical_port_returns, risk_free_rate=0.047,initial_value=10000):
	"""
	Produce four-pnel summary visual of the full analysis
	"""

	fig,axes=plt.subplots(2,2,figsize=(18,12))
	fig.suptitle('Portfolio Optimization Analysis', fontsize=16,fontweight='bold')

	#Normalized price history
	ax1 = axes[0,0]
	(prices/prices.iloc[0]*100).plot(ax=ax1)
	ax1.set_title('Panel 1: Normalized Price History (Base 100)')
	ax1.set_xlabel('Date')
	ax1.set_ylabel('Indexed Price')
	ax1.legend(fontsize=8)
	ax1.grid(True)

	#Correlation map
	ax2 = axes[0,1]
	sns.heatmap(log_returns.corr(), annot=True,fmt='.2f',cmap='coolwarm',center=0, ax=ax2)
	ax2.set_title("Panel 2: Return Correlation Matrix")

	#Efficient frontier
	ax3=axes[1,0]
	min_ret = np.dot(min_weights,mean_returns)
	min_vol = np.sqrt(np.dot(min_weights.T,np.dot(cov_matrix,min_weights)))
	max_ret = np.dot(max_weights,mean_returns)
	max_vol = np.sqrt(np.dot(max_weights.T,np.dot(cov_matrix,max_weights)))
	max_sharpe =(max_ret-risk_free_rate)/max_vol

	ax3.plot(frontier_vols,frontier_returns,'-b',linewidth=2,label="Efficient Frontier")
	ax3.scatter(min_vol,min_ret,marker="*",color='red',s=300,zorder=5,label="Min Variance")
	ax3.scatter(max_vol,max_ret,marker="*",color="gold",s=300,zorder=5,label=f"Max Sharpe ({max_sharpe:.2f})")
	cml_x=np.linspace(0,max_vol *1.5,100)
	cml_y=risk_free_rate + max_sharpe * cml_x
	ax3.plot(cml_x,cml_y,"r--",linewidth=1.5,label="Capital Market Line")
	ax3.set_title("Panel 3: Efficient Frontier")
	ax3.set_xlabel("Annualized Volatility")
	ax3.set_ylabel("Annualized Expected Return")
	ax3.legend(fontsize=8)
	ax3.grid(True)

	#Monte Carlo
	ax4=axes[1,1]
	ax4.plot(simulation_results,color="lightblue",alpha=0.5,linewidth=0.5)
	ax4.plot(np.percentile(simulation_results,50,axis=1),color='blue',linewidth=2,label="Median")
	ax4.plot(np.percentile(simulation_results,5,axis=1),color="red",linewidth=2,label="5th percentile")
	ax4.plot(np.percentile(simulation_results,95,axis=1),color="orange",linewidth=2,label="95th percentile")
	ax4.axhline(y=initial_value,color='black',linestyle="--",linewidth=1)
	ax4.set_title("Panel 4: Monte Carlo Simulation (1 Year Forward)")
	ax4.set_xlabel('Trading Days')
	ax4.set_ylabel("Portfolio Value ($)")
	ax4.legend(fontsize=8)
	ax4.grid(True)

	plt.tight_layout()
	plt.savefig('portfolio_analysis_summary.png',dpi=150,bbox_inches="tight")
	plt.show()
	print("Chart saved as portfolio_analysis_summary.png")





