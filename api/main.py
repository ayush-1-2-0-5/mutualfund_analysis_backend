from flask import Flask, request, jsonify
import requests
import pandas as pd
from indiafactorlibrary import IndiaFactorLibrary
import warnings
from sklearn.metrics import r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

warnings.filterwarnings("ignore")

app = Flask(__name__)

class GetNAVSeriesAndFactorData:
    def __init__(self):
        self.ifl = IndiaFactorLibrary()
        self.fund_id = None
        self.nav_series = None
        self.nav_series_monthly = None
        self.nav_series_yearly = None
        self.dataset = None

    def fetch_nav_data(self, fund_id):
        url = f"https://api.mfapi.in/mf/{fund_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            dates = [entry['date'] for entry in data['data']]
            navs = [float(entry['nav']) for entry in data['data']]
            self.nav_series = pd.Series(navs, index=pd.to_datetime(dates, format='%d-%m-%Y'))
            return self.nav_series.astype(str).to_dict()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching NAV data: {e}")

    def display_dataset(self, factor_id):
        try:
            self.dataset = self.ifl.read(factor_id)
            return "Dataset loaded successfully."
        except Exception as e:
            raise Exception(f"Error displaying the dataset: {e}")

    def monthly_returns(self, nav_series_new=None):
        nav_series = nav_series_new if nav_series_new is not None else self.nav_series
        nav_series = nav_series.sort_index(ascending=True)
        ratios = pd.Series(index=nav_series.index, dtype='float64')
        ratios[nav_series.index[0]] = 0

        for i in range(1, len(nav_series)):
            ratios[nav_series.index[i]] = ((nav_series.iloc[i] - nav_series.iloc[i - 1]) / nav_series.iloc[i - 1])

        monthly_sums = ratios.groupby(ratios.index.to_period('M')).sum()
        return monthly_sums

    def set_monthly_returns_in_data(self, index):
        data = self.dataset.get(index)
        if data is None:
            raise Exception(f"No data found for index: {index}")

        monthly_returns = self.monthly_returns()
        data['monthly_returns'] = data.index.to_period('M').map(monthly_returns) * 100
        data = data.dropna()
        self.dataset[index] = data

    def yearly_returns(self, nav_series_new=None, index=None):
        if index is None:
            raise Exception("Index must be provided to access the dataset.")

        data = self.dataset.get(index)
        if data is None:
            raise Exception(f"No data found for index: {index}")

        if self.nav_series_monthly is None:
            self.nav_series_monthly = self.monthly_returns()

        monthly_nav_series = self.nav_series_monthly.sort_index(ascending=True)
        yearly_returns = pd.Series(index=monthly_nav_series.index.year.unique(), dtype='float64')
        yearly_grouped = monthly_nav_series.groupby(monthly_nav_series.index.year)

        for year, group in yearly_grouped:
            if len(group) == 12:
                yearly_return = 1
                for month_return in group:
                    yearly_return *= (1 + month_return)
                yearly_return -= 1
                yearly_returns.at[year] = yearly_return * 100

        data['year'] = data.index.year
        data = data.join(yearly_returns.rename('yearly_return'), on='year', how='inner')
        data = data.drop(columns=['year']).dropna()
        self.dataset[index] = data

    def parse_options(self, description):
        options_dict = {}
        lines = description.strip().split('\n')
        descs = {}
        for line in lines:
            key, rest = line.split(':', 1)
            key = int(key.strip())
            parts = rest.strip().split('--')
            desc = parts[0].strip()
            descs[desc] = parts[1].strip().split()[0].lower() if len(parts) > 1 else 'unknown'

            options_dict[key] = {
                "type": descs[desc],
                "description": desc
            }
        return options_dict

    def get_return_frequency(self):
        if 'DESCR' not in self.dataset:
            raise Exception("Dataset description is not available.")

        descr = self.dataset['DESCR']
        descr_parts = descr.split("\n\n")
        options_dict = self.parse_options(descr_parts[-1])

        for option in options_dict:
            if len(self.dataset[option]) < 50:
                options_dict[option]['type'] = 'annually'
            else:
                options_dict[option]['type'] = 'monthly'

        return options_dict

    def process_return_frequency(self, user_input):
        options_dict = self.get_return_frequency()
        if user_input not in options_dict:
            raise Exception(f"Invalid input. Please enter a number between 0 and {len(options_dict) - 1}.")

        if options_dict[user_input]['type'] == 'monthly':
            self.set_monthly_returns_in_data(index=user_input)
        else:
            self.yearly_returns(index=user_input)

        return self.dataset[user_input]

nav_data = GetNAVSeriesAndFactorData()

@app.route('/search_fund', methods=['GET', 'POST'])
def search_fund():
    if request.method == 'POST':
        query = request.json.get('query')
    else:  # GET request
        query = request.args.get('query')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    url = f"https://api.mfapi.in/mf/search?q={query}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        funds = response.json()
        return jsonify(funds)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 400

@app.route('/fetch_nav_data', methods=['POST'])
def fetch_nav_data():
    fund_id = request.json.get('fund_id')
    if not fund_id:
        return jsonify({"error": "Fund ID is required"}), 400

    try:
        nav_data.fetch_nav_data(fund_id)
        return jsonify({"message": "NAV data fetched successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/available_factors', methods=['GET'])
def available_factors():
    try:
        available_datasets = nav_data.ifl.get_available_datasets()
        return jsonify(available_datasets)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/display_dataset', methods=['POST'])
def display_dataset():
    factor_id = request.json.get('factor_id')
    if not factor_id:
        return jsonify({"error": "Factor ID is required"}), 400

    try:
        message = nav_data.display_dataset(factor_id)
        return jsonify({"message": message, "description": nav_data.dataset['DESCR']})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/get_return_frequency', methods=['GET'])
def get_return_frequency():
    try:
        options_dict = nav_data.get_return_frequency()
        return jsonify(options_dict)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/process_return_frequency', methods=['POST'])
def process_return_frequency():
    user_input = request.json.get('user_input')
    if user_input is None:
        return jsonify({"error": "User input is required"}), 400

    try:
        user_input = int(user_input)
    except ValueError:
        return jsonify({"error": "User input must be a valid integer"}), 400

    try:
        options_dict = nav_data.get_return_frequency()
        if user_input not in options_dict:
            return jsonify({"error": f"Invalid input. Please enter a number between 0 and {len(options_dict) - 1}."}), 400

        processed_data = nav_data.process_return_frequency(user_input)

        # Convert to a JSON-friendly format
        if isinstance(processed_data, pd.DataFrame):
            processed_data = processed_data.reset_index().to_dict(orient='records')
        else:
            processed_data = processed_data.to_dict()

        return jsonify(processed_data)
    except Exception as e: 
        return jsonify({"error": str(e)}), 400

@app.route('/perform_multiregression', methods=['POST'])
def perform_multiregression():
    data = request.json
    target_column = data.get('target_column')
    dataset_index = data.get('dataset_index')

    # Validate required fields
    if target_column is None or dataset_index is None:
        return jsonify({"error": "Target column and dataset index are required"}), 400

    try:
      
        dataframe = nav_data.dataset[dataset_index]

        if target_column not in dataframe.columns:
            return jsonify({"error": f"Target column '{target_column}' not found in the dataset."}), 400
        selected_columns = [col for col in dataframe.columns if col != target_column]
        X = dataframe[selected_columns]
        y = dataframe[target_column]
        X = sm.add_constant(X)  
        model = sm.OLS(y, X).fit()
        summary_json = {
            'rsquared': model.rsquared,
            'params': model.params.to_dict(),
            'pvalues': model.pvalues.to_dict(),
        }
        plt.figure(figsize=(15, 10))
        sns.heatmap(dataframe[selected_columns + [target_column]].corr(), annot=True, cmap="YlGnBu")
        plt.title('Correlation Matrix')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({
            'summary': summary_json,
            'r2_score': r2_score(y, model.predict(X)),
            'correlation_plot': plot_base64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run()

