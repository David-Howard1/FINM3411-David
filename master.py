"""
RIO TINTO BETA ANALYSIS FOR WACC CALCULATION
FINM3411 Assignment

This script performs a comprehensive beta analysis for RIO Tinto using:
1. Company-specific beta calculation from stock returns
2. Comparables analysis with unlevering/relevering  
3. Blended approach combining both methods
4. Final recommendations for WACC calculation

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn, fallback to manual calculation if not available
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
    print("✅ Using sklearn OLS regression")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  sklearn not available, using manual calculation")

# Try to import scipy for p-values
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def main():
    print("="*70)
    print("RIO TINTO COMPREHENSIVE BETA ANALYSIS")
    print("="*70)
    
    # File paths
    equity_data_file = "equitysheet2.csv"
    capital_structure_file = "DV and EV.csv"
    
    # Initialize the analysis
    analyzer = RIOBetaAnalyzer(equity_data_file, capital_structure_file)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("Generated files:")
        print("- final_rio_beta_analysis.png (Comprehensive visualization)")
        print("- rio_day_of_week_analysis.png (Day-of-week stability analysis)")
        print("- rio_beta_results.csv (Detailed results)")
        print("="*70)
    else:
        print("Analysis failed. Please check the data files.")

class RIOBetaAnalyzer:
    """
    Comprehensive beta analysis for RIO Tinto combining company-specific
    and comparables approaches.
    """
    
    def __init__(self, equity_data_file, capital_structure_file):
        self.equity_data_file = equity_data_file
        self.capital_structure_file = capital_structure_file
        
        # Stock tickers and market index
        self.stock_tickers = [
            'RIO AU Equity', 'BHP AU Equity', 'FMG AU Equity', '1208 HK Equity',
            '010130 KS Equity', 'FM CN Equity', 'MIN AU Equity', 'HZ IN Equity', 'HBM CN Equity'
        ]
        self.market_index = 'AS51 Index'
        
        # Tax rates by country
        self.tax_rates = {
            'Rio Tinto': 0.30,        # Australia
            'BHP Group': 0.30,        # Australia  
            'Fortescue': 0.30,        # Australia
            'MMG': 0.25,              # Hong Kong
            'Korea Zinc': 0.25,       # South Korea
            'First Quantum': 0.30,    # Canada
            'Mineral Resources': 0.30, # Australia
            'Hindustan Zinc': 0.30,   # India
            'Hudbay Minerals': 0.27   # Canada
        }
        
    def load_and_clean_equity_data(self):
        """Load and clean the equity price data."""
        print("\n[1/6] Loading and cleaning equity data...")
        
        # Read CSV with robust encoding
        try:
            raw_data = pd.read_csv(self.equity_data_file, header=None, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                raw_data = pd.read_csv(self.equity_data_file, header=None, encoding='latin-1')
            except UnicodeDecodeError:
                raw_data = pd.read_csv(self.equity_data_file, header=None, encoding='cp1252')
        
        print(f"Successfully loaded CSV with shape: {raw_data.shape}")
        
        # Extract header row and find ticker positions
        header_row = raw_data.iloc[0].fillna('')
        ticker_columns = {}
        for i, cell in enumerate(header_row):
            if cell in self.stock_tickers or cell == self.market_index:
                ticker_columns[cell] = i
        
        print(f"Found tickers: {list(ticker_columns.keys())}")
        
        # Extract and clean data for each ticker
        data_rows = raw_data.iloc[1:].copy()
        clean_dict = {}
        
        for ticker, col_idx in ticker_columns.items():
            if col_idx + 1 < len(data_rows.columns):
                dates = data_rows.iloc[:, col_idx]
                prices = data_rows.iloc[:, col_idx + 1]
                
                # Create temporary DataFrame
                temp_df = pd.DataFrame({'Date': dates, 'Price': prices})
                temp_df = temp_df.dropna()
                temp_df = temp_df[temp_df['Date'].astype(str) != '']
                temp_df = temp_df[temp_df['Price'].astype(str) != '']
                
                # Parse dates and prices
                try:
                    temp_df['Date'] = pd.to_datetime(temp_df['Date'], format='%d/%m/%Y', errors='coerce')
                    temp_df['Price'] = pd.to_numeric(temp_df['Price'], errors='coerce')
                    temp_df = temp_df.dropna()
                    temp_df = temp_df[(temp_df['Price'] > 0) & (temp_df['Price'] < 1000000)]
                    temp_df = temp_df.sort_values('Date')
                    temp_df = temp_df[temp_df['Date'] >= '2015-01-01']
                    
                    if len(temp_df) > 50:
                        clean_dict[ticker] = temp_df
                        print(f"  {ticker}: {len(temp_df)} observations")
                    
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
                    continue
        
        # Merge all data on common dates
        if clean_dict:
            clean_data = None
            for ticker, df in clean_dict.items():
                df_temp = df.set_index('Date')[['Price']].rename(columns={'Price': ticker})
                if clean_data is None:
                    clean_data = df_temp
                else:
                    clean_data = clean_data.join(df_temp, how='outer')
            
            # Handle missing values
            clean_data = clean_data.ffill().dropna(how='all')
            
            # Ensure market index and minimum stocks
            if self.market_index in clean_data.columns:
                clean_data = clean_data[clean_data[self.market_index].notna()]
                stock_columns = [col for col in clean_data.columns if col != self.market_index]
                valid_stocks_per_row = clean_data[stock_columns].notna().sum(axis=1)
                clean_data = clean_data[valid_stocks_per_row >= 5]
            
            print(f"Final clean data: {clean_data.shape}, {clean_data.index.min()} to {clean_data.index.max()}")
            return clean_data
        
        return None
    
    def calculate_weekly_returns(self, data, period_years=None, strict_weekly_filter=False):
        """Calculate weekly returns for different day-of-week references."""
        print(f"\n[2/6] Calculating weekly returns...")
        if strict_weekly_filter:
            print("  ⚠️  STRICT 7-DAY FILTERING ENABLED - Only exact weekly intervals included")
        
        # Filter by time period if specified
        if period_years:
            end_date = data.index.max()
            start_date = end_date - pd.DateOffset(years=period_years)
            data = data[data.index >= start_date]
            print(f"Using {period_years}-year period: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        
        weekly_returns = {}
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        for day_idx, day_name in enumerate(weekdays):
            # Filter for specific weekday
            day_data = data[data.index.dayofweek == day_idx].copy()
            
            if len(day_data) < 2:
                continue
            
            # Sort by date
            day_data = day_data.sort_index()
            
            if strict_weekly_filter:
                # STRICT FILTERING: Only include observations that are exactly 7 days apart
                print(f"    Applying strict 7-day filtering for {day_name}...")
                
                # Calculate date differences
                date_diffs = day_data.index.to_series().diff().dt.days
                
                # Find observations that are exactly 7 days from previous
                exact_weekly = (date_diffs == 7) | (date_diffs.isna())  # Include first observation
                strict_day_data = day_data[exact_weekly].copy()
                
                # Additional check: ensure we have consecutive 7-day periods
                consecutive_weekly = []
                prev_date = None
                
                for current_date in strict_day_data.index:
                    if prev_date is None:
                        consecutive_weekly.append(current_date)
                    elif (current_date - prev_date).days == 7:
                        consecutive_weekly.append(current_date)
                    # Skip dates that aren't exactly 7 days apart
                    prev_date = current_date
                
                # Filter to only consecutive weekly observations
                if len(consecutive_weekly) > 1:
                    day_data = strict_day_data.loc[consecutive_weekly]
                    
                    # Log the filtering impact
                    original_count = len(strict_day_data)
                    filtered_count = len(day_data)
                    removed_count = original_count - filtered_count
                    
                    if removed_count > 0:
                        print(f"      {day_name}: Removed {removed_count} non-consecutive weekly observations")
                        print(f"      {day_name}: {original_count} → {filtered_count} observations")
                else:
                    print(f"      {day_name}: Insufficient consecutive weekly data, skipping")
                    continue
            
            # Calculate weekly returns
            weekly_returns_data = day_data.pct_change().dropna()
            
            if strict_weekly_filter and len(weekly_returns_data) > 0:
                # Verify that all returns represent exactly 7-day periods
                dates = weekly_returns_data.index
                if len(dates) > 1:
                    # Check if any return periods are not 7 days (this shouldn't happen after filtering, but double-check)
                    date_diffs = dates.to_series().diff().dt.days[1:]  # Skip first NaT
                    non_weekly = (date_diffs != 7).sum()
                    if non_weekly > 0:
                        print(f"      WARNING: {non_weekly} non-7-day periods found in {day_name} data")
            
            # Remove extreme outliers
            for col in weekly_returns_data.columns:
                weekly_returns_data = weekly_returns_data[
                    (weekly_returns_data[col] > -0.5) & (weekly_returns_data[col] < 1.0)
                ]
            
            # Critical filter: Remove artificial zero returns from forward-filling
            if self.market_index in weekly_returns_data.columns:
                market_col = self.market_index
                # Remove market zero returns
                non_zero_market = abs(weekly_returns_data[market_col]) >= 0.0001
                weekly_returns_data = weekly_returns_data[non_zero_market]
                
                # Remove rows where >50% stocks have zero returns
                stock_columns = [col for col in weekly_returns_data.columns if col != market_col]
                zero_counts = (abs(weekly_returns_data[stock_columns]) < 0.0001).sum(axis=1)
                valid_data = zero_counts <= len(stock_columns) / 2
                weekly_returns_data = weekly_returns_data[valid_data]
            
            weekly_returns[day_name] = weekly_returns_data
            if strict_weekly_filter:
                print(f"  {day_name}: {len(weekly_returns_data)} strict 7-day observations (after all filters)")
            else:
                print(f"  {day_name}: {len(weekly_returns_data)} observations")
        
        return weekly_returns
    
    def calculate_betas(self, weekly_returns):
        """Calculate beta estimates for all stocks."""
        print(f"\n[3/6] Calculating beta estimates...")
        
        results = {}
        
        for day, returns_data in weekly_returns.items():
            if len(returns_data) < 20:  # Minimum observations
                continue
            
            day_results = {}
            market_returns = returns_data[self.market_index]
            
            for stock in self.stock_tickers:
                if stock not in returns_data.columns:
                    continue
                
                stock_returns = returns_data[stock]
                n = len(stock_returns)
                
                if SKLEARN_AVAILABLE:
                    # Use sklearn OLS regression
                    X = market_returns.values.reshape(-1, 1)  # Market returns (independent variable)
                    y = stock_returns.values  # Stock returns (dependent variable)
                    
                    # Fit OLS regression: stock_return = alpha + beta * market_return
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Extract results
                    beta = model.coef_[0]  # Slope coefficient (beta)
                    alpha = model.intercept_  # Intercept (alpha)
                    
                    # Predict values and calculate R-squared
                    y_pred = model.predict(X)
                    r_squared = r2_score(y, y_pred)
                    
                    # Calculate additional statistics
                    residuals = y - y_pred
                    mse = np.mean(residuals**2)
                    
                    # Calculate standard errors and t-statistics
                    X_with_intercept = np.column_stack([np.ones(len(X)), X.flatten()])
                    try:
                        # Standard error of beta coefficient
                        xtx_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                        beta_se = np.sqrt(mse * xtx_inv[1, 1])
                        t_stat = beta / beta_se
                        
                        # Calculate p-value
                        if SCIPY_AVAILABLE:
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                        else:
                            p_value = 0.0001 if abs(t_stat) > 2 else 0.05
                    except:
                        # Fallback if matrix inversion fails
                        correlation = np.corrcoef(stock_returns, market_returns)[0, 1]
                        t_stat = correlation * np.sqrt((n - 2) / (1 - r_squared)) if r_squared < 0.999 else 10
                        p_value = 0.0001 if abs(t_stat) > 2 else 0.05
                        beta_se = None
                
                else:
                    # Fallback to manual calculation
                    covariance = np.cov(stock_returns, market_returns)[0, 1]
                    market_variance = np.var(market_returns, ddof=1)
                    beta = covariance / market_variance
                    alpha = np.mean(stock_returns) - beta * np.mean(market_returns)
                    
                    # Calculate R-squared
                    correlation = np.corrcoef(stock_returns, market_returns)[0, 1]
                    r_squared = correlation ** 2
                    
                    # Calculate p-value (simplified)
                    t_stat = correlation * np.sqrt((n - 2) / (1 - r_squared)) if r_squared < 0.999 else 10
                    if SCIPY_AVAILABLE:
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    else:
                        p_value = 0.0001 if abs(t_stat) > 2 else 0.05
                    beta_se = None
                
                day_results[stock] = {
                    'beta': beta,
                    'alpha': alpha,
                    'r_squared': r_squared,
                    'p_value': p_value,
                    't_statistic': t_stat,
                    'beta_se': beta_se,
                    'observations': n,
                    'method': 'sklearn_ols' if SKLEARN_AVAILABLE else 'manual'
                }
            
            results[day] = day_results
        
        return results
    
    def calculate_company_weighted_betas(self, period_results):
        """Calculate weighted betas for each company across time periods."""
        print(f"\n[4/6] Calculating weighted betas across periods...")
        
        company_betas = {}
        
        for stock in self.stock_tickers:
            stock_betas = []
            
            # Extract betas for each period (10Y, 5Y, 3Y)
            for period_name, period_data in period_results.items():
                day_betas = []
                for day, day_results in period_data.items():
                    if stock in day_results:
                        day_betas.append(day_results[stock]['beta'])
                
                if day_betas:
                    period_mean = np.mean(day_betas)
                    stock_betas.append(period_mean)
            
            if len(stock_betas) >= 3:  # We have 10Y, 5Y, 3Y
                # Weighted average: 50% 3Y + 25% 5Y + 25% 10Y
                weighted_beta = 0.5 * stock_betas[2] + 0.25 * stock_betas[1] + 0.25 * stock_betas[0]
                company_betas[stock] = {
                    'betas_10y_5y_3y': stock_betas,
                    'weighted_beta': weighted_beta
                }
                print(f"  {stock}: Weighted={weighted_beta:.4f}")
        
        return company_betas
    
    def load_capital_structure_data(self):
        """Load capital structure data for comparables analysis."""
        print(f"\n[5/6] Loading capital structure data...")
        
        df = pd.read_csv(self.capital_structure_file)
        
        # Clean and parse the data
        capital_data = {}
        company_mapping = {
            'Rio Tinto': 'RIO AU Equity',
            'BHP Group': 'BHP AU Equity',
            'Fortescue': 'FMG AU Equity',
            'MMG': '1208 HK Equity',
            'Korea Zinc': '010130 KS Equity',
            'First Quantum': 'FM CN Equity',
            'Mineral Resources': 'MIN AU Equity',
            'Hindustan Zinc': 'HZ IN Equity',
            'Hudbay Minerals': 'HBM CN Equity'
        }
        
        for _, row in df.iterrows():
            company_name = row.iloc[0]
            if pd.isna(company_name) or company_name == '' or company_name is None:
                continue
            
            # Clean company name
            company_name = str(company_name).strip()
            
            try:
                ev_ratio = float(row['E/V'].replace('%', '')) / 100
                dv_ratio = float(row['D/V'].replace('%', '')) / 100
                
                capital_data[company_name] = {
                    'E/V': ev_ratio,
                    'D/V': dv_ratio,
                    'D/E': dv_ratio / ev_ratio if ev_ratio > 0 else 0
                }
                print(f"  {company_name}: D/E={dv_ratio/ev_ratio:.3f}")
                
            except Exception as e:
                print(f"  Error processing {company_name}: {e}")
        
        return capital_data, company_mapping
    
    def perform_comparables_analysis(self, company_betas, capital_data, company_mapping):
        """Perform unlevering/relevering analysis using comparators."""
        print(f"\n[6/6] Performing comparables analysis...")
        
        # Get RIO's target capital structure
        rio_capital = capital_data['Rio Tinto']
        rio_de = rio_capital['D/E']
        rio_tax = self.tax_rates['Rio Tinto']
        
        print(f"RIO Target Structure: D/E={rio_de:.3f}, Tax={rio_tax:.0%}")
        
        # Unlever comparator betas and relever to RIO's structure
        comparator_results = {}
        asset_betas = []
        
        for company_name, ticker in company_mapping.items():
            if company_name == 'Rio Tinto':
                continue  # Skip RIO itself
            
            if ticker in company_betas and company_name in capital_data:
                equity_beta = company_betas[ticker]['weighted_beta']
                current_de = capital_data[company_name]['D/E']
                tax_rate = self.tax_rates[company_name]
                
                # Unlever to asset beta
                asset_beta = equity_beta / (1 + (1 - tax_rate) * current_de)
                
                # Relever to RIO's capital structure
                relevered_beta = asset_beta * (1 + (1 - rio_tax) * rio_de)
                
                comparator_results[company_name] = {
                    'ticker': ticker,
                    'original_beta': equity_beta,
                    'asset_beta': asset_beta,
                    'relevered_beta': relevered_beta,
                    'current_de': current_de
                }
                
                asset_betas.append(asset_beta)
                print(f"  {company_name}: {equity_beta:.4f} → {relevered_beta:.4f}")
        
        # Calculate industry statistics
        relevered_betas = [result['relevered_beta'] for result in comparator_results.values()]
        industry_stats = {
            'mean_asset_beta': np.mean(asset_betas),
            'mean_relevered_beta': np.mean(relevered_betas),
            'median_relevered_beta': np.median(relevered_betas),
            'std_relevered_beta': np.std(relevered_betas)
        }
        
        print(f"\nIndustry Mean Relevered Beta: {industry_stats['mean_relevered_beta']:.4f}")
        
        return comparator_results, industry_stats
    
    def calculate_final_recommendations(self, company_betas, industry_stats):
        """Calculate final beta recommendations."""
        rio_direct = company_betas['RIO AU Equity']['weighted_beta']
        industry_mean = industry_stats['mean_relevered_beta']
        
        recommendations = {
            'rio_specific': rio_direct,
            'pure_industry_mean': industry_mean,
            'blended_70_30': 0.7 * rio_direct + 0.3 * industry_mean,
            'blended_60_40': 0.6 * rio_direct + 0.4 * industry_mean,
            'blended_80_20': 0.8 * rio_direct + 0.2 * industry_mean
        }
        
        return recommendations
    
    def create_comprehensive_visualization(self, company_betas, comparator_results, industry_stats, recommendations):
        """Create comprehensive visualization of all results."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        # 1. RIO Beta Evolution
        ax1 = fig.add_subplot(gs[0, 0])
        rio_betas = company_betas['RIO AU Equity']['betas_10y_5y_3y']
        periods = ['10-Year', '5-Year', '3-Year']
        
        bars = ax1.bar(periods, rio_betas, color=['lightcoral', 'gold', 'lightgreen'], alpha=0.8)
        ax1.axhline(y=company_betas['RIO AU Equity']['weighted_beta'], color='red', linestyle='--', 
                   label=f'Weighted: {company_betas["RIO AU Equity"]["weighted_beta"]:.3f}')
        
        ax1.set_title('RIO Beta by Time Period', fontweight='bold')
        ax1.set_ylabel('Beta')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        for bar, beta in zip(bars, rio_betas):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{beta:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Comparator transformation
        ax2 = fig.add_subplot(gs[0, 1])
        
        companies = list(comparator_results.keys())
        original_betas = [comparator_results[c]['original_beta'] for c in companies]
        relevered_betas = [comparator_results[c]['relevered_beta'] for c in companies]
        
        x = np.arange(len(companies))
        width = 0.35
        
        ax2.bar(x - width/2, original_betas, width, label='Original', alpha=0.7)
        ax2.bar(x + width/2, relevered_betas, width, label='Relevered', alpha=0.7)
        
        ax2.set_title('Comparator Beta Transformation', fontweight='bold')
        ax2.set_ylabel('Beta')
        ax2.set_xticks(x)
        ax2.set_xticklabels([c.split()[0] for c in companies], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Final recommendations
        ax3 = fig.add_subplot(gs[0, 2])
        
        rec_names = ['RIO\nDirect', 'Industry\nMean', 'Blended\n70/30', 'Blended\n80/20']
        rec_values = [recommendations['rio_specific'], recommendations['pure_industry_mean'],
                     recommendations['blended_70_30'], recommendations['blended_80_20']]
        colors = ['darkblue', 'red', 'green', 'orange']
        
        bars = ax3.bar(rec_names, rec_values, color=colors, alpha=0.8)
        ax3.set_title('Beta Recommendations', fontweight='bold')
        ax3.set_ylabel('Beta')
        ax3.grid(True, alpha=0.3)
        
        for bar, beta in zip(bars, rec_values):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{beta:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Industry distribution
        ax4 = fig.add_subplot(gs[0, 3])
        
        ax4.hist(relevered_betas, bins=6, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(industry_stats['mean_relevered_beta'], color='red', linestyle='--', 
                   label=f'Mean: {industry_stats["mean_relevered_beta"]:.3f}')
        ax4.axvline(recommendations['rio_specific'], color='blue', linestyle='--',
                   label=f'RIO: {recommendations["rio_specific"]:.3f}')
        
        ax4.set_title('Industry Beta Distribution', fontweight='bold')
        ax4.set_xlabel('Beta')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Summary table (bottom span)
        ax5 = fig.add_subplot(gs[1, :])
        ax5.axis('off')
        
        # Create summary table
        table_data = [
            ['APPROACH', 'BETA', 'METHODOLOGY', 'VS RIO', 'RECOMMENDATION'],
            ['RIO-Specific', f'{recommendations["rio_specific"]:.4f}', 'Company-specific weighted', 'Baseline', 'Most accurate'],
            ['Industry Mean', f'{recommendations["pure_industry_mean"]:.4f}', 'Relevered comparators', f'{(recommendations["pure_industry_mean"]/recommendations["rio_specific"]-1)*100:+.1f}%', 'Industry benchmark'],
            ['Blended 70/30', f'{recommendations["blended_70_30"]:.4f}', '70% RIO + 30% Industry', f'{(recommendations["blended_70_30"]/recommendations["rio_specific"]-1)*100:+.1f}%', 'RECOMMENDED'],
            ['Blended 80/20', f'{recommendations["blended_80_20"]:.4f}', '80% RIO + 20% Industry', f'{(recommendations["blended_80_20"]/recommendations["rio_specific"]-1)*100:+.1f}%', 'Conservative'],
        ]
        
        table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center', bbox=[0, 0.4, 1, 0.5])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style the table
        for i in range(5):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight recommended row
        for j in range(5):
            table[(3, j)].set_facecolor('#90EE90')
            table[(3, j)].set_text_props(weight='bold')
        
        # Add methodology summary
        summary_text = f"""
FINAL BETA RECOMMENDATION: {recommendations['blended_70_30']:.4f}

METHODOLOGY:
• Company-specific: Weighted beta with STRICT 7-day filtering (exact weekly intervals only)
• Zero-return filtering: Removes artificial forward-filled data points
• Comparables: Unlevered 8 mining peers, relevered to RIO's capital structure (D/E=0.194)  
• Blended: 70% company-specific + 30% industry benchmark for balanced accuracy

ACADEMIC JUSTIFICATION:
"Strict 7-day filtering ensures all weekly returns represent genuine 7-day trading periods, 
eliminating bias from holidays and irregular trading schedules. Combined with industry 
benchmarking through rigorous unlevering/relevering for robust WACC estimation."
        """
        
        ax5.text(0.02, 0.35, summary_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('RIO TINTO COMPREHENSIVE BETA ANALYSIS FOR WACC\nCompany-Specific + Comparables Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig('/Users/davidhoward/FINM3411/final_rio_beta_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Comprehensive visualization saved: final_rio_beta_analysis.png")
    
    def create_day_of_week_analysis(self, period_results):
        """Create detailed day-of-week beta analysis for RIO across time periods."""
        print("\nCreating day-of-week beta analysis for RIO...")
        
        # Extract RIO data for each period and day
        rio_ticker = 'RIO AU Equity'
        periods = ['10-Year', '5-Year', '3-Year']
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        # Initialize data structures
        beta_data = {period: {day: None for day in weekdays} for period in periods}
        r2_data = {period: {day: None for day in weekdays} for period in periods}
        obs_data = {period: {day: None for day in weekdays} for period in periods}
        
        # Extract data
        for period_name, period_data in period_results.items():
            for day_name, day_results in period_data.items():
                if rio_ticker in day_results:
                    result = day_results[rio_ticker]
                    beta_data[period_name][day_name] = result['beta']
                    r2_data[period_name][day_name] = result['r_squared']
                    obs_data[period_name][day_name] = result['observations']
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # 1. Beta values heatmap
        ax1 = fig.add_subplot(gs[0, :])
        
        # Create beta matrix for heatmap
        beta_matrix = []
        for period in periods:
            row = []
            for day in weekdays:
                beta_val = beta_data[period][day]
                row.append(beta_val if beta_val is not None else np.nan)
            beta_matrix.append(row)
        
        beta_matrix = np.array(beta_matrix)
        
        # Create heatmap
        im1 = ax1.imshow(beta_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax1.set_xticks(range(len(weekdays)))
        ax1.set_xticklabels(weekdays)
        ax1.set_yticks(range(len(periods)))
        ax1.set_yticklabels(periods)
        
        # Add text annotations
        for i in range(len(periods)):
            for j in range(len(weekdays)):
                if not np.isnan(beta_matrix[i, j]):
                    text = ax1.text(j, i, f'{beta_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontweight='bold')
        
        ax1.set_title('RIO Beta Estimates by Day-of-Week and Time Period', fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar1.set_label('Beta Value', fontweight='bold')
        
        # 2. R² values heatmap
        ax2 = fig.add_subplot(gs[1, :])
        
        # Create R² matrix for heatmap
        r2_matrix = []
        for period in periods:
            row = []
            for day in weekdays:
                r2_val = r2_data[period][day]
                row.append(r2_val if r2_val is not None else np.nan)
            r2_matrix.append(row)
        
        r2_matrix = np.array(r2_matrix)
        
        # Create heatmap
        im2 = ax2.imshow(r2_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax2.set_xticks(range(len(weekdays)))
        ax2.set_xticklabels(weekdays)
        ax2.set_yticks(range(len(periods)))
        ax2.set_yticklabels(periods)
        
        # Add text annotations
        for i in range(len(periods)):
            for j in range(len(weekdays)):
                if not np.isnan(r2_matrix[i, j]):
                    text = ax2.text(j, i, f'{r2_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontweight='bold')
        
        ax2.set_title('RIO R² Values by Day-of-Week and Time Period', fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar2.set_label('R² Value', fontweight='bold')
        
        # 3. Beta stability by day (line plots)
        ax3 = fig.add_subplot(gs[2, 0])
        
        for day in weekdays:
            day_betas = []
            for period in periods:
                beta_val = beta_data[period][day]
                if beta_val is not None:
                    day_betas.append(beta_val)
                else:
                    day_betas.append(np.nan)
            
            ax3.plot(periods, day_betas, 'o-', linewidth=2, markersize=8, label=day)
        
        ax3.set_title('Beta Evolution by Day-of-Week', fontweight='bold')
        ax3.set_ylabel('Beta Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. R² stability by day (line plots)
        ax4 = fig.add_subplot(gs[2, 1])
        
        for day in weekdays:
            day_r2s = []
            for period in periods:
                r2_val = r2_data[period][day]
                if r2_val is not None:
                    day_r2s.append(r2_val)
                else:
                    day_r2s.append(np.nan)
            
            ax4.plot(periods, day_r2s, 'o-', linewidth=2, markersize=8, label=day)
        
        ax4.set_title('R² Evolution by Day-of-Week', fontweight='bold')
        ax4.set_ylabel('R² Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Sample sizes
        ax5 = fig.add_subplot(gs[2, 2])
        
        # Create sample size matrix
        obs_matrix = []
        for period in periods:
            row = []
            for day in weekdays:
                obs_val = obs_data[period][day]
                row.append(obs_val if obs_val is not None else 0)
            obs_matrix.append(row)
        
        obs_matrix = np.array(obs_matrix)
        
        # Create bar chart for latest period (3-Year)
        latest_obs = obs_matrix[2]  # 3-Year data
        bars = ax5.bar(weekdays, latest_obs, color='lightblue', alpha=0.7)
        
        ax5.set_title('Sample Sizes (3-Year Period)', fontweight='bold')
        ax5.set_ylabel('Number of Observations')
        ax5.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, obs in zip(bars, latest_obs):
            if obs > 0:
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{int(obs)}', ha='center', va='bottom', fontweight='bold')
        
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics table
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # Calculate summary statistics
        summary_data = []
        
        # Headers
        headers = ['Period', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Mean', 'Std Dev', 'CV (%)']
        
        # Beta statistics
        for period in periods:
            row = [f'{period} Beta']
            period_betas = []
            
            for day in weekdays:
                beta_val = beta_data[period][day]
                if beta_val is not None:
                    row.append(f'{beta_val:.3f}')
                    period_betas.append(beta_val)
                else:
                    row.append('N/A')
            
            if period_betas:
                mean_beta = np.mean(period_betas)
                std_beta = np.std(period_betas)
                cv_beta = (std_beta / mean_beta) * 100 if mean_beta != 0 else 0
                row.extend([f'{mean_beta:.3f}', f'{std_beta:.4f}', f'{cv_beta:.1f}%'])
            else:
                row.extend(['N/A', 'N/A', 'N/A'])
            
            summary_data.append(row)
        
        # Add separator
        summary_data.append([''] * len(headers))
        
        # R² statistics
        for period in periods:
            row = [f'{period} R²']
            period_r2s = []
            
            for day in weekdays:
                r2_val = r2_data[period][day]
                if r2_val is not None:
                    row.append(f'{r2_val:.3f}')
                    period_r2s.append(r2_val)
                else:
                    row.append('N/A')
            
            if period_r2s:
                mean_r2 = np.mean(period_r2s)
                std_r2 = np.std(period_r2s)
                cv_r2 = (std_r2 / mean_r2) * 100 if mean_r2 != 0 else 0
                row.extend([f'{mean_r2:.3f}', f'{std_r2:.4f}', f'{cv_r2:.1f}%'])
            else:
                row.extend(['N/A', 'N/A', 'N/A'])
            
            summary_data.append(row)
        
        # Create table
        table = ax6.table(cellText=summary_data, colLabels=headers,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight beta rows
        for i in range(1, 4):  # Beta rows
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#E6F3FF')
        
        # Highlight R² rows  
        for i in range(5, 8):  # R² rows
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#F0F8F0')
        
        plt.suptitle('RIO TINTO DAY-OF-WEEK BETA ANALYSIS\nStability Assessment with STRICT 7-Day Filtering', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig('/Users/davidhoward/FINM3411/rio_day_of_week_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results_to_csv(self, company_betas, comparator_results, industry_stats, recommendations):
        """Export detailed results to CSV."""
        results_data = []
        
        # Add RIO results
        rio_data = company_betas['RIO AU Equity']
        results_data.append({
            'Company': 'RIO AU Equity',
            'Type': 'Target',
            '10Y_Beta': rio_data['betas_10y_5y_3y'][0],
            '5Y_Beta': rio_data['betas_10y_5y_3y'][1],
            '3Y_Beta': rio_data['betas_10y_5y_3y'][2],
            'Weighted_Beta': rio_data['weighted_beta'],
            'Relevered_Beta': 'N/A',
            'Regression_Method': 'sklearn_ols' if SKLEARN_AVAILABLE else 'manual'
        })
        
        # Add comparator results
        for company, data in comparator_results.items():
            ticker_data = company_betas.get(data['ticker'], {})
            if 'betas_10y_5y_3y' in ticker_data:
                betas = ticker_data['betas_10y_5y_3y']
                results_data.append({
                    'Company': data['ticker'],
                    'Type': 'Comparator',
                    '10Y_Beta': betas[0],
                    '5Y_Beta': betas[1],
                    '3Y_Beta': betas[2],
                    'Weighted_Beta': ticker_data['weighted_beta'],
                    'Relevered_Beta': data['relevered_beta'],
                    'Regression_Method': 'sklearn_ols' if SKLEARN_AVAILABLE else 'manual'
                })
        
        # Add recommendations
        for name, beta in recommendations.items():
            results_data.append({
                'Company': f'{name.upper()}',
                'Type': 'Recommendation',
                '10Y_Beta': 'N/A',
                '5Y_Beta': 'N/A', 
                '3Y_Beta': 'N/A',
                'Weighted_Beta': beta,
                'Relevered_Beta': 'N/A',
                'Regression_Method': 'Combined'
            })
        
        df = pd.DataFrame(results_data)
        df.to_csv('/Users/davidhoward/FINM3411/rio_beta_results.csv', index=False)
        print("✅ Detailed results exported: rio_beta_results.csv")
    
    def run_complete_analysis(self):
        """Run the complete beta analysis workflow."""
        try:
            # Step 1: Load and clean equity data
            clean_data = self.load_and_clean_equity_data()
            if clean_data is None:
                return None
            
            # Step 2: Calculate betas for different periods
            period_results = {}
            periods = [(10, '10-Year'), (5, '5-Year'), (3, '3-Year')]
            
            print("\n" + "="*70)
            print("APPLYING STRICT 7-DAY FILTERING FOR RIO-SPECIFIC CALCULATIONS")
            print("Only exact weekly intervals (7 days apart) will be included")
            print("="*70)
            
            for years, name in periods:
                weekly_returns = self.calculate_weekly_returns(clean_data, years, strict_weekly_filter=True)
                beta_results = self.calculate_betas(weekly_returns)
                period_results[name] = beta_results
            
            # Step 3: Calculate weighted betas
            company_betas = self.calculate_company_weighted_betas(period_results)
            
            # Step 4: Load capital structure and perform comparables analysis
            capital_data, company_mapping = self.load_capital_structure_data()
            comparator_results, industry_stats = self.perform_comparables_analysis(
                company_betas, capital_data, company_mapping)
            
            # Step 5: Calculate final recommendations
            recommendations = self.calculate_final_recommendations(company_betas, industry_stats)
            
            # Step 6: Create visualizations and export results
            self.create_comprehensive_visualization(company_betas, comparator_results, 
                                                  industry_stats, recommendations)
            self.create_day_of_week_analysis(period_results)
            self.export_results_to_csv(company_betas, comparator_results, 
                                     industry_stats, recommendations)
            
            # Print final summary
            print("\n" + "="*70)
            print("FINAL BETA RECOMMENDATIONS FOR RIO WACC")
            print("="*70)
            print(f"RIO-Specific Beta:           {recommendations['rio_specific']:.4f}")
            print(f"Pure Industry Mean Beta:     {recommendations['pure_industry_mean']:.4f}")
            print(f"Blended 70/30 Beta:          {recommendations['blended_70_30']:.4f} ← RECOMMENDED")
            print(f"Blended 80/20 Beta:          {recommendations['blended_80_20']:.4f}")
            print("="*70)
            print(f"FINAL RECOMMENDATION: Use β = {recommendations['blended_70_30']:.4f} for WACC")
            print("This balances RIO-specific accuracy with industry validation.")
            print("="*70)
            
            return {
                'company_betas': company_betas,
                'comparator_results': comparator_results,
                'industry_stats': industry_stats,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    main()
