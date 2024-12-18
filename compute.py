import numpy as np
import ot

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def compute_cost_matrix(bin_edges):
    """
    Compute the cost matrix based on the squared Euclidean distance
    between bin centers.

    Parameters:
        bin_edges (ndarray): Edges of the bins used for histograms.

    Returns:
        cost_matrix (ndarray): A cost matrix based on squared distances between bin centers.
    """
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cost_matrix = np.square(bin_centers[:, None] - bin_centers[None, :])
    return cost_matrix

def compute_barycenter(stocks, startdate, enddate, period, symbols, bins=50, reg=0.01, numItermax=1000):
    histograms = {}
    all_returns = []
    for symbol, df in stocks.items():
        if symbol not in symbols:
            continue
        df = df.sort_index() # ensure the DatetimeIndex is sorted
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"Index for {symbol} is not sorted correctly.")
        time_period_data = df.loc[startdate:enddate]
        if period not in time_period_data.columns:
            print(f"{period} not found in {symbol} data frame. skipping")
            continue
        returns = time_period_data[period].dropna()
        all_returns.extend(returns)

    if len(all_returns) == 0:
        raise ValueError("No valid data found for the specified symbols and time range.")
    
    all_returns = np.array(all_returns)
    common_bin_edges = np.linspace(all_returns.min(), all_returns.max(), bins + 1)

    for symbol, df in stocks.items():
        if symbol not in symbols:
            continue
        df = df.sort_index() # ensure the DatetimeIndex is sorted
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"Index for {symbol} is not sorted correctly.")
        time_period_data = df.loc[startdate:enddate]
        returns = time_period_data[period].dropna()
        hist, _ = np.histogram(returns, bins=common_bin_edges, density=True)
        histograms[symbol] = hist


    A = np.array(list(histograms.values())).T
    barycenter = ot.bregman.barycenter(
        A,
        M=compute_cost_matrix(common_bin_edges),
        reg=reg,
        weights=np.ones(len(histograms)) / len(histograms),  # Uniform weights
        numItermax=numItermax,
    )

    return barycenter, histograms, common_bin_edges
    

def visualize_barycenter_seaborn(barycenter, histograms, bin_edges):
    """
    Visualize true stock histograms as bar plots and overlay the barycenter as a density line.
    
    Parameters:
        barycenter (ndarray): The computed barycenter.
        histograms (dict): Histograms for each stock.
        bin_edges (ndarray): The shared bin edges used for histograms.
        symbols (list): List of stock symbols.
    """
    # Compute bin centers for the barycenter
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    plt.figure(figsize=(12, 6))
    
    # Plot histograms as bar plots
    for symbol, hist in histograms.items():
        plt.bar(
            bin_centers, 
            hist, 
            width=(bin_edges[1] - bin_edges[0]),
            # width=(bin_edges[1] - bin_edges[0]) * 0.9,  # Slightly reduce bar width for visual clarity
            alpha=0.4,
            label=f"{symbol} Histogram"
        )
    
    # Plot the barycenter as a density line
    sns.lineplot(x=bin_centers, y=barycenter, color='black', label="Barycenter", linewidth=2)
    
    # Add labels, title, and legend
    plt.title("Stock Return Histograms with Barycenter")
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def denormalize_data(normalized_values, scaler):
    """
    Denormalize data using the inverse transformation of the scaler.

    Parameters:
        normalized_values (ndarray): Normalized values (e.g., barycenter or histogram bins).
        scaler (StandardScaler): The scaler used to normalize the data.

    Returns:
        ndarray: Denormalized values.
    """
    return normalized_values * scaler.scale_[0] + scaler.mean_[0]

def visualize_raw_barycenter(barycenter, histograms, bin_edges, scaler):
    """
    Visualize raw stock histograms as bar plots and overlay the raw barycenter as a density line.

    Parameters:
        barycenter (ndarray): Normalized barycenter.
        histograms (dict): Normalized histograms for each stock.
        bin_edges (ndarray): Shared bin edges in the normalized domain.
        scaler (StandardScaler): Scaler used for normalization.
        symbols (list): List of stock symbols.
    """
    # Denormalize bin edges to get raw bin centers
    bin_centers_norm = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers_raw = denormalize_data(bin_centers_norm, scaler)
    
    # Denormalize barycenter
    barycenter_raw = denormalize_data(barycenter, scaler)
    
    plt.figure(figsize=(12, 6))
    
    # Plot histograms for each stock in the raw domain
    for symbol, hist in histograms.items():
        hist_raw = hist / np.sum(hist)  # Ensure the histogram is normalized to density
        plt.bar(
            bin_centers_raw, 
            hist_raw, 
            width=(bin_centers_raw[1] - bin_centers_raw[0]) * 0.9, 
            alpha=0.4,
            label=f"{symbol} Histogram"
        )
    
    # Overlay barycenter as a density line
    barycenter_raw = barycenter_raw / barycenter_raw.sum()
    sns.lineplot(x=bin_centers_raw, y=barycenter_raw, color='black', label="Barycenter", linewidth=2)
    
    # Add labels, title, and legend
    plt.title("Raw Stock Return Distributions with Barycenter")
    plt.xlabel("Return (Raw)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def visualize_barycenter_as_lines_raw(barycenter, histograms, bin_edges, scaler, xlim, alpha=0.3):
    """
    Visualize the barycenter and stock distributions as smooth lines.

    Parameters:
        barycenter (ndarray): The computed barycenter.
        histograms (dict): Histograms for each stock.
        bin_edges (ndarray): The shared bin edges used for histograms.
    """
    # Compute bin centers for plotting
    # Denormalize bin edges to get raw bin centers
    bin_centers_norm = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers_raw = denormalize_data(bin_centers_norm, scaler)
    
    # Denormalize barycenter
    barycenter_raw = denormalize_data(barycenter, scaler)
    
    plt.figure(figsize=(12, 6))
    
    # Plot stock distributions as smooth lines with alpha=0.3
    for symbol, hist in histograms.items():
        # hist = hist / hist.sum()
        sns.lineplot(x=bin_centers_raw, y=hist, label=f"{symbol} Distribution", alpha=alpha)
    
    # Plot the barycenter as a bold black line
    # barycenter_raw = barycenter_raw / barycenter_raw.sum()
    sns.lineplot(x=bin_centers_raw, y=barycenter, color='black', label="Barycenter", linewidth=2)
    
    # Add labels, title, and legend
    plt.title("Stock Return Distributions with Barycenter")
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.xlim(xlim)
    plt.legend()
    plt.show()

def compute_wasserstein_distances_pot(histograms, barycenter, bin_edges):
    """
    Compute the Wasserstein distance between stock density histograms and the barycenter using POT.

    Parameters:
        histograms (dict): Dictionary of normalized histograms for each stock.
        barycenter (ndarray): The computed barycenter (normalized or raw).
        bin_edges (ndarray): The bin edges corresponding to the histograms.

    Returns:
        distances (dict): Dictionary with stock symbols as keys and Wasserstein distances as values.
    """
    distances = {}
    
    # Compute bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Compute cost matrix based on squared Euclidean distances between bin centers
    cost_matrix = ot.utils.dist(bin_centers.reshape(-1, 1), bin_centers.reshape(-1, 1), metric='sqeuclidean')
    
    for symbol, hist in histograms.items():
        # Ensure histograms are properly normalized to sum to 1
        hist = hist / np.sum(hist)
        barycenter = barycenter / np.sum(barycenter)
        
        # Compute Wasserstein distance using POT
        wasserstein_distance = ot.emd2(hist, barycenter, cost_matrix)
        distances[symbol] = np.sqrt(wasserstein_distance)  # Convert from squared Wasserstein to regular distance
    
    return distances

def compute_expected_return_and_risk(histogram, bin_edges, normalize=True):
    """
    Compute expected return and risk, ensuring histogram normalization if needed.

    Parameters:
        histogram (ndarray): Density histogram (raw or normalized).
        bin_edges (ndarray): Bin edges corresponding to the histogram.
        normalize (bool): Whether to normalize the histogram (default: True).

    Returns:
        expected_return (float): The computed expected return.
        risk (float): The computed standard deviation (risk).
    """
    # Normalize histogram if required
    if normalize:
        histogram = histogram / np.sum(histogram)
    
    # Compute bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Compute expected return
    expected_return = np.sum(histogram * bin_centers)
    
    # Compute variance and risk
    variance = np.sum(histogram * (bin_centers - expected_return) ** 2)
    risk = np.sqrt(variance)
    
    return expected_return, risk

def visualize_barycenter_as_lines(barycenter, histograms, bin_edges, xlim, alpha=0.3):
    """
    Visualize the barycenter and stock distributions as smooth lines.

    Parameters:
        barycenter (ndarray): The computed barycenter.
        histograms (dict): Histograms for each stock.
        bin_edges (ndarray): The shared bin edges used for histograms.
    """
    # Compute bin centers for plotting
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    plt.figure(figsize=(12, 6))
    
    # Plot stock distributions as smooth lines with alpha=0.3
    for symbol, hist in histograms.items():
        # sns.kdeplot(x=bin_centers, y=hist, label=f"{symbol} Distribution", alpha=alpha)
        sns.lineplot(x=bin_centers, y=hist, label=f"{symbol} Distribution", alpha=alpha)
    
    # Plot the barycenter as a bold black line
    sns.lineplot(x=bin_centers, y=barycenter, color='black', label="Barycenter", linewidth=2)
    
    # Add labels, title, and legend
    plt.title("Stock Return Distributions with Barycenter")
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.xlim(xlim)
    plt.legend()
    plt.show()


def visualize_barycenter_as_smooth_lineplot(barycenter, histograms, bin_edges, xlim, alpha=0.3, spline_degree=3, num_points=500):
    """
    Visualize the barycenter and stock distributions as smooth lines using spline interpolation.

    Parameters:
        barycenter (ndarray): The computed barycenter.
        histograms (dict): Histograms for each stock.
        bin_edges (ndarray): The shared bin edges used for histograms.
        xlim (tuple): Limits for the x-axis.
        alpha (float): Transparency level for the stock distributions.
        spline_degree (int): Degree of the spline (default is cubic).
        num_points (int): Number of points for the smooth curve.
    """
    # Compute bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Generate a finer grid for smooth curves
    x_new = np.linspace(bin_centers.min(), bin_centers.max(), num_points)
    
    plt.figure(figsize=(12, 6))
    
    # Plot each histogram as a smooth spline
    for symbol, hist in histograms.items():
        spline = make_interp_spline(bin_centers, hist, k=spline_degree)
        hist_smooth = spline(x_new)
        sns.lineplot(x=x_new, y=hist_smooth, label=f"{symbol}", alpha=alpha)
    
    # Smooth the barycenter
    spline_bary = make_interp_spline(bin_centers, barycenter, k=spline_degree)
    bary_smooth = spline_bary(x_new)
    sns.lineplot(x=x_new, y=bary_smooth, color='black', label="Barycenter", linewidth=2)
    
    # Add labels, title, and legend
    # plt.title("Stock Return Distributions with Barycenter")
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.xlim(xlim)
    plt.legend()
    plt.show()
    

def denormalize_return_and_risk(normalized_return, normalized_risk, scaler):
    """
    Denormalize expected return and risk to their original scale.

    Parameters:
        normalized_return (float): Expected return in normalized space.
        normalized_risk (float): Expected risk (std dev) in normalized space.
        scaler (StandardScaler): The scaler fitted on all returns.

    Returns:
        denormalized_return (float): Expected return in raw space.
        denormalized_risk (float): Expected risk (std dev) in raw space.
    """
    mean = scaler.mean_[0]
    std_dev = scaler.scale_[0]
    
    denormalized_return = normalized_return * std_dev + mean
    denormalized_risk = normalized_risk * std_dev
    return denormalized_return, denormalized_risk


def compute_returns_and_risk_for_stocks(data, return_type='daily_return', bins=50):
    """
    Compute the expected return and risk for each stock in the dataset.

    Parameters:
        data (dict): Dictionary containing stock DataFrames.
        return_type (str): Column name for the type of return ('daily_return', etc.).
        bins (int): Number of bins for the histogram.

    Returns:
        results (dict): Dictionary of results with expected return and risk for each stock.
    """
    results = {}

    for symbol, df in data.items():
        if return_type not in df.columns:
            print(f"{return_type} not found in {symbol}. Skipping.")
            continue

        # Drop NaN values
        returns = df[return_type].dropna()
        
        # Compute histogram
        hist, bin_edges = np.histogram(returns, bins=bins, density=True)
        
        # Compute expected return and risk
        expected_return, risk = compute_expected_return_and_risk(hist, bin_edges)
        
        # Store results
        results[symbol] = {'Expected Return': expected_return, 'Risk': risk}
    
    return results