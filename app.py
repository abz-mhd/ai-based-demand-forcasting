# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AI Inventory Management",
    page_icon="📊",
    layout="wide"
)

# Constants
LOW_STOCK_THRESHOLD = 40
RESTOCK_THRESHOLD = 25
PRODUCTS_PER_PAGE = 10
DATA_FILE = "inventory_data.json"

class DemandForecastingApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.products_df = None
        self.load_models()
        self.load_products_data()
        
        # Initialize session state with persistence
        self.initialize_session_state()
    
    def save_data(self):
        """Save all data to JSON file"""
        try:
            data_to_save = {
                'sales_data': st.session_state.sales_data,
                'products': st.session_state.products,
                'restock_history': st.session_state.restock_history,
                'show_ai_predictions': st.session_state.show_ai_predictions
            }
            
            # Convert datetime objects to strings for JSON serialization
            for sale in data_to_save['sales_data']:
                if 'timestamp' in sale and isinstance(sale['timestamp'], datetime):
                    sale['timestamp'] = sale['timestamp'].isoformat()
            
            for restock in data_to_save['restock_history']:
                if 'timestamp' in restock and isinstance(restock['timestamp'], datetime):
                    restock['timestamp'] = restock['timestamp'].isoformat()
            
            with open(DATA_FILE, 'w') as f:
                json.dump(data_to_save, f, indent=4)
        except Exception as e:
            st.error(f"Error saving data: {e}")
    
    def load_data(self):
        """Load data from JSON file"""
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r') as f:
                    data = json.load(f)
                
                # Convert string timestamps back to datetime objects
                for sale in data.get('sales_data', []):
                    if 'timestamp' in sale and isinstance(sale['timestamp'], str):
                        sale['timestamp'] = datetime.fromisoformat(sale['timestamp'])
                
                for restock in data.get('restock_history', []):
                    if 'timestamp' in restock and isinstance(restock['timestamp'], str):
                        restock['timestamp'] = datetime.fromisoformat(restock['timestamp'])
                
                return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
        return None
    
    def initialize_session_state(self):
        """Initialize session state with persistent data"""
        if 'initialized' not in st.session_state:
            # Try to load existing data first
            saved_data = self.load_data()
            
            if saved_data:
                # Use saved data
                st.session_state.sales_data = saved_data.get('sales_data', [])
                st.session_state.products = saved_data.get('products', self.get_products_dict())
                st.session_state.restock_history = saved_data.get('restock_history', [])
                st.session_state.show_ai_predictions = saved_data.get('show_ai_predictions', False)
            else:
                # Initialize with default data
                st.session_state.sales_data = []
                st.session_state.products = self.get_products_dict()
                st.session_state.restock_history = []
                st.session_state.show_ai_predictions = False
            
            # Initialize other session state variables
            st.session_state.initialized = True
            st.session_state.shop_page = 0
            st.session_state.admin_page = 0
            st.session_state.forecast_page = 0
            st.session_state.sales_forecast_data = {}
            st.session_state.chart_counter = 0
            st.session_state.selected_product_id = None
    
    def get_unique_key(self, prefix="chart"):
        """Generate unique keys for Streamlit elements"""
        st.session_state.chart_counter += 1
        return f"{prefix}_{st.session_state.chart_counter}"
    
    def load_models(self):
        """Load trained models"""
        try:
            self.model = joblib.load('xgboost_demand_forecasting_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoders = joblib.load('label_encoders.pkl')
            self.feature_columns = joblib.load('feature_names.pkl')
        except FileNotFoundError as e:
            st.sidebar.info("🤖 AI Model: Not loaded")
    
    def load_products_data(self):
        """Load products dataset"""
        try:
            dataset_path = os.path.join('dataset', 'retail_store_inventory.csv')
            self.products_df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            st.error("Dataset not found at: dataset/retail_store_inventory.csv")
            self.products_df = pd.DataFrame()
    
    def get_products_dict(self):
        """Convert dataframe to products dictionary"""
        if self.products_df.empty:
            return {}
        
        products_dict = {}
        
        # Find actual column names
        column_mapping = {
            'product_id': ['Product_ID', 'product_id', 'ID', 'SKU'],
            'name': ['Product_Name', 'product_name', 'Name'],
            'category': ['Category', 'category', 'Type'],
            'price': ['Price', 'price', 'Unit_Price'],
            'stock': ['Stock_Quantity', 'stock', 'Quantity', 'Current_Stock'],
        }
        
        actual_columns = {}
        for standard_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in self.products_df.columns:
                    actual_columns[standard_name] = possible_name
                    break
        
        for index, row in self.products_df.iterrows():
            product_id = row[actual_columns.get('product_id', self.products_df.columns[0])] if actual_columns.get('product_id') else f"P{index:04d}"
            name = row[actual_columns.get('name', self.products_df.columns[1])] if actual_columns.get('name') else f'Product {product_id}'
            category = row[actual_columns.get('category', self.products_df.columns[2])] if actual_columns.get('category') else 'General'
            
            try:
                price = float(row[actual_columns.get('price', self.products_df.columns[3])]) if actual_columns.get('price') else 50.0
            except:
                price = 50.0
            
            try:
                stock = int(row[actual_columns.get('stock', self.products_df.columns[4])]) if actual_columns.get('stock') else 100
            except:
                stock = 100
            
            products_dict[str(product_id)] = {
                'name': str(name),
                'category': str(category),
                'price': price,
                'current_stock': stock,
                'original_stock': stock
            }
        
        return products_dict
    
    def get_basic_stats(self):
        """Get basic statistics"""
        products = st.session_state.products
        sales_data = st.session_state.sales_data
        
        total_products = len(products)
        total_sales = len(sales_data)
        total_revenue = sum(sale['total'] for sale in sales_data)
        low_stock_count = len([p for p in products.values() if p['current_stock'] <= RESTOCK_THRESHOLD])
        out_of_stock_count = len([p for p in products.values() if p['current_stock'] <= 0])
        
        return {
            'total_products': total_products,
            'total_sales': total_sales,
            'total_revenue': total_revenue,
            'low_stock_count': low_stock_count,
            'out_of_stock_count': out_of_stock_count
        }
    
    def get_sales_products(self):
        """Get products that have sales history"""
        if not st.session_state.sales_data:
            return {}
        
        # Get unique product IDs from sales data
        sold_product_ids = set(sale['product_id'] for sale in st.session_state.sales_data)
        
        # Return only products that have sales history
        sales_products = {pid: details for pid, details in st.session_state.products.items() 
                         if pid in sold_product_ids}
        
        return sales_products
    
    def analyze_sales_patterns(self, product_id):
        """Analyze sales patterns for a product"""
        if not st.session_state.sales_data:
            return None
        
        sales_df = pd.DataFrame(st.session_state.sales_data)
        product_sales = sales_df[sales_df['product_id'] == product_id].copy()
        
        if len(product_sales) == 0:
            return None
        
        # Convert timestamp to datetime and extract features
        product_sales['date'] = pd.to_datetime(product_sales['timestamp'])
        product_sales['day_of_week'] = product_sales['date'].dt.dayofweek
        product_sales['month'] = product_sales['date'].dt.month
        product_sales['week'] = product_sales['date'].dt.isocalendar().week
        
        # Calculate sales metrics
        total_sales = len(product_sales)
        total_quantity = product_sales['quantity'].sum()
        avg_daily_sales = total_quantity / max(1, (product_sales['date'].max() - product_sales['date'].min()).days)
        
        # Weekly pattern
        weekly_sales = product_sales.groupby('day_of_week')['quantity'].sum()
        
        # Monthly pattern
        monthly_sales = product_sales.groupby('month')['quantity'].sum()
        
        # Recent trend (last 30 days vs previous 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_sales = product_sales[product_sales['date'] >= recent_cutoff]['quantity'].sum()
        older_sales = product_sales[product_sales['date'] < recent_cutoff]['quantity'].sum()
        
        trend = "Stable"
        if recent_sales > older_sales * 1.2:
            trend = "Growing"
        elif recent_sales < older_sales * 0.8:
            trend = "Declining"
        
        return {
            'total_sales': total_sales,
            'total_quantity': total_quantity,
            'avg_daily_sales': avg_daily_sales,
            'weekly_pattern': weekly_sales.to_dict(),
            'monthly_pattern': monthly_sales.to_dict(),
            'trend': trend,
            'recent_sales': recent_sales,
            'sales_history_days': (product_sales['date'].max() - product_sales['date'].min()).days
        }
    
    def generate_sales_forecast(self, product_id, product_details):
        """Generate 3-month sales forecast based on historical patterns"""
        try:
            # Analyze sales patterns
            sales_patterns = self.analyze_sales_patterns(product_id)
            if not sales_patterns:
                return None
            
            current_stock = product_details['current_stock']
            avg_daily_sales = sales_patterns['avg_daily_sales']
            trend = sales_patterns['trend']
            
            # Base monthly forecast from average daily sales
            base_monthly = avg_daily_sales * 30
            
            # Apply trend factor
            trend_factors = {
                "Growing": 1.2,
                "Declining": 0.8,
                "Stable": 1.0
            }
            trend_factor = trend_factors.get(trend, 1.0)
            
            # Generate 3-month forecast with seasonal variations
            monthly_forecasts = {}
            current_date = datetime.now()
            total_forecast = 0
            
            for month_offset in range(1, 4):  # Next 3 months
                forecast_date = current_date + timedelta(days=30 * month_offset)
                month_name = forecast_date.strftime("%B %Y")
                
                # Apply seasonal factors
                seasonal_factor = self.get_seasonal_factor(forecast_date.month)
                
                # Apply monthly pattern if available
                monthly_pattern_factor = 1.0
                if forecast_date.month in sales_patterns['monthly_pattern']:
                    avg_monthly = np.mean(list(sales_patterns['monthly_pattern'].values()))
                    if avg_monthly > 0:
                        monthly_pattern_factor = sales_patterns['monthly_pattern'][forecast_date.month] / avg_monthly
                
                # Calculate monthly forecast
                monthly_forecast = base_monthly * trend_factor * seasonal_factor * monthly_pattern_factor
                monthly_forecast = max(5, monthly_forecast)  # Minimum forecast
                
                monthly_forecasts[month_name] = {
                    'demand': round(monthly_forecast),
                    'seasonal_factor': seasonal_factor,
                    'trend_factor': trend_factor,
                    'monthly_pattern_factor': monthly_pattern_factor,
                    'confidence': self.calculate_confidence(sales_patterns)
                }
                
                total_forecast += monthly_forecast
            
            # Calculate stock coverage
            coverage_months = current_stock / (total_forecast / 3) if total_forecast > 0 else float('inf')
            
            # Generate risk assessment
            if coverage_months < 1:
                risk_level = "High"
                recommendation = "🚨 URGENT: Stock may run out within forecast period"
            elif coverage_months < 2:
                risk_level = "Medium"
                recommendation = "⚠️ WARNING: Monitor stock levels closely"
            else:
                risk_level = "Low"
                recommendation = "✅ STABLE: Adequate stock for forecast period"
            
            return {
                'monthly_forecasts': monthly_forecasts,
                'total_3month_forecast': total_forecast,
                'sales_patterns': sales_patterns,
                'current_stock': current_stock,
                'coverage_months': coverage_months,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'product_name': product_details['name'],
                'category': product_details['category'],
                'confidence': self.calculate_confidence(sales_patterns)
            }
            
        except Exception as e:
            return None
    
    def calculate_confidence(self, sales_patterns):
        """Calculate forecast confidence based on sales data quality"""
        if not sales_patterns:
            return "Low"
        
        confidence_factors = []
        
        # Sales history duration
        if sales_patterns['sales_history_days'] > 90:
            confidence_factors.append(1.0)
        elif sales_patterns['sales_history_days'] > 30:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # Number of sales transactions
        if sales_patterns['total_sales'] > 50:
            confidence_factors.append(1.0)
        elif sales_patterns['total_sales'] > 20:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Data consistency
        if len(sales_patterns['monthly_pattern']) >= 3:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        avg_confidence = np.mean(confidence_factors)
        
        if avg_confidence >= 0.8:
            return "High"
        elif avg_confidence >= 0.6:
            return "Medium"
        else:
            return "Low"
    
    def get_seasonal_factor(self, month):
        """Get seasonal adjustment factor based on month"""
        seasonal_factors = {
            1: 1.1, 2: 0.9, 3: 1.0, 4: 1.2, 5: 1.1, 6: 1.3,
            7: 1.4, 8: 1.2, 9: 1.0, 10: 1.1, 11: 1.3, 12: 1.5
        }
        return seasonal_factors.get(month, 1.0)
    
    def generate_sales_forecasts_all(self):
        """Generate 3-month forecasts for all products with sales history"""
        sales_products = self.get_sales_products()
        forecasts = {}
        
        for pid, details in sales_products.items():
            forecast = self.generate_sales_forecast(pid, details)
            if forecast:
                forecasts[pid] = forecast
        
        # Store forecasts in session state
        st.session_state.sales_forecast_data = forecasts
        return forecasts
    
    def permanent_restock(self, product_id, quantity, reason="Manual Restock"):
        """Permanently restock a product and save data"""
        if product_id in st.session_state.products:
            st.session_state.products[product_id]['current_stock'] += quantity
            
            restock_record = {
                'timestamp': datetime.now(),
                'product_id': product_id,
                'product_name': st.session_state.products[product_id]['name'],
                'quantity_added': quantity,
                'new_stock_level': st.session_state.products[product_id]['current_stock'],
                'reason': reason
            }
            st.session_state.restock_history.append(restock_record)
            
            # Save data after restocking
            self.save_data()
            return True
        return False
    
    def buy_product(self, product_id, quantity):
        """Handle product purchase and save data"""
        if product_id in st.session_state.products:
            current_stock = st.session_state.products[product_id]['current_stock']
            
            if quantity <= current_stock:
                st.session_state.products[product_id]['current_stock'] -= quantity
                
                sale_record = {
                    'timestamp': datetime.now(),
                    'product_id': product_id,
                    'product_name': st.session_state.products[product_id]['name'],
                    'quantity': quantity,
                    'price': st.session_state.products[product_id]['price'],
                    'total': quantity * st.session_state.products[product_id]['price']
                }
                st.session_state.sales_data.append(sale_record)
                
                if not st.session_state.show_ai_predictions:
                    st.session_state.show_ai_predictions = True
                
                # Save data after purchase
                self.save_data()
                return True
        return False

def create_sales_forecast_chart(forecast_data, unique_key):
    """Create animated sales forecast chart"""
    if not forecast_data or 'monthly_forecasts' not in forecast_data:
        return None
    
    months = list(forecast_data['monthly_forecasts'].keys())
    demands = [forecast_data['monthly_forecasts'][month]['demand'] for month in months]
    confidences = [forecast_data['monthly_forecasts'][month]['confidence'] for month in months]
    
    # Color based on confidence
    colors = []
    for conf in confidences:
        if conf == "High":
            colors.append('#4CAF50')  # Green
        elif conf == "Medium":
            colors.append('#FFA726')  # Orange
        else:
            colors.append('#FF6B6B')  # Red
    
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=months,
        y=demands,
        name="Sales Forecast",
        marker_color=colors,
        text=demands,
        textposition='outside',
        texttemplate='%{y:.0f} units',
        hovertemplate='<b>%{x}</b><br>Forecast: %{y} units<br>Confidence: %{customdata}',
        customdata=confidences
    ))
    
    fig.update_layout(
        title="📈 3-Month Sales Forecast",
        xaxis_title="Forecast Period",
        yaxis_title="Predicted Sales (Units)",
        height=400,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def create_sales_trend_analysis(forecast_data, unique_key):
    """Create sales trend analysis chart"""
    if not forecast_data or 'sales_patterns' not in forecast_data:
        return None
    
    patterns = forecast_data['sales_patterns']
    
    # Weekly pattern
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_sales = [patterns['weekly_pattern'].get(i, 0) for i in range(7)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=days,
        y=weekly_sales,
        mode='lines+markers',
        name="Weekly Sales Pattern",
        line=dict(color='#2196F3', width=3),
        marker=dict(size=8, color='#2196F3')
    ))
    
    fig.update_layout(
        title="📊 Weekly Sales Pattern",
        xaxis_title="Day of Week",
        yaxis_title="Total Sales (Units)",
        height=350,
        template="plotly_white"
    )
    
    return fig

def create_sales_products_overview(forecasts):
    """Create overview of products with sales forecasts"""
    if not forecasts:
        return None
    
    # Summary statistics
    total_products = len(forecasts)
    high_confidence = len([f for f in forecasts.values() if f['confidence'] == 'High'])
    high_risk = len([f for f in forecasts.values() if f['risk_level'] == 'High'])
    total_forecast = sum(f['total_3month_forecast'] for f in forecasts.values())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Products with Sales", total_products)
    
    with col2:
        st.metric("🎯 High Confidence", high_confidence)
    
    with col3:
        st.metric("🚨 High Risk", high_risk)
    
    with col4:
        st.metric("📈 Total 3-Month Forecast", f"{total_forecast:.0f} units")

def create_risk_analysis_chart(forecasts, unique_key):
    """Create risk analysis chart"""
    if not forecasts:
        return None
    
    risk_levels = {}
    for forecast in forecasts.values():
        risk = forecast['risk_level']
        risk_levels[risk] = risk_levels.get(risk, 0) + 1
    
    colors = {'High': '#FF6B6B', 'Medium': '#FFA726', 'Low': '#4CAF50'}
    
    fig = px.pie(
        values=list(risk_levels.values()),
        names=list(risk_levels.keys()),
        title="🎯 Forecast Risk Distribution",
        color=list(risk_levels.keys()),
        color_discrete_map=colors
    )
    
    fig.update_layout(height=400)
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    return fig

def display_sales_forecast_card(pid, details, forecast_data, app):
    """Display sales forecast card - PRODUCTS ON TOP"""
    st.markdown("---")
    
    # Product information at the TOP
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"### {details['name']}")
        st.write(f"**Category:** {details['category']}")
        st.write(f"**Current Stock:** {forecast_data['current_stock']} units")
        st.write(f"**Sales History:** {forecast_data['sales_patterns']['total_sales']} transactions")
        st.write(f"**Trend:** {forecast_data['sales_patterns']['trend']}")
    
    with col2:
        # Risk and recommendation
        risk_level = forecast_data['risk_level']
        recommendation = forecast_data['recommendation']
        
        if risk_level == "High":
            st.error(f"**{recommendation}**")
        elif risk_level == "Medium":
            st.warning(f"**{recommendation}**")
        else:
            st.success(f"**{recommendation}**")
        
        # Coverage information
        st.info(f"**Stock Coverage:** {forecast_data['coverage_months']:.1f} months")
        
        # Confidence information (text only, no gauge)
        confidence_level = forecast_data['confidence']
        confidence_color = {
            "High": "green",
            "Medium": "orange", 
            "Low": "red"
        }.get(confidence_level, "gray")
        
        st.markdown(f"**Forecast Confidence:** :{confidence_color}[**{confidence_level}**]")
        
        # Restock action for high risk
        if risk_level in ["High", "Medium"]:
            suggested_qty = max(50, forecast_data['total_3month_forecast'] * 0.3)
            # Ensure suggested quantity doesn't exceed reasonable limits
            suggested_qty = min(suggested_qty, 5000)
            
            restock_amount = st.number_input(
                "Restock amount",
                min_value=1,
                max_value=10000,
                value=int(suggested_qty),
                key=f"sales_restock_{pid}"
            )
            
            if st.button(f"🔄 Restock Based on Forecast", key=f"sales_btn_{pid}"):
                if app.permanent_restock(pid, restock_amount, "Sales Forecast Recommended"):
                    st.success(f"✅ Restocked {restock_amount} units!")
                    st.rerun()
    
    # Monthly forecasts in a separate section
    st.subheader("📅 Monthly Forecasts")
    monthly_data = forecast_data['monthly_forecasts']
    month_cols = st.columns(3)
    for idx, (month_name, month_forecast) in enumerate(monthly_data.items()):
        with month_cols[idx]:
            confidence_color = {
                "High": "green",
                "Medium": "orange",
                "Low": "red"
            }.get(month_forecast['confidence'], "gray")
            
            st.metric(
                f"{month_name}",
                f"{month_forecast['demand']} units",
                f"Confidence: {month_forecast['confidence']}"
            )
    
    # GRAPHS BELOW THE PRODUCT INFO
    st.subheader("📊 Sales Analysis Charts")
    
    # Create two columns for charts - REMOVED CONFIDENCE GAUGE
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sales forecast chart
        forecast_key = app.get_unique_key(f"forecast_{pid}")
        forecast_chart = create_sales_forecast_chart(forecast_data, forecast_key)
        if forecast_chart:
            st.plotly_chart(forecast_chart, use_container_width=True, key=forecast_key)
    
    with col2:
        # Empty column (previously had confidence gauge)
        st.write("")  # Empty space for layout balance
    
    # Weekly trend chart - full width
    trend_key = app.get_unique_key(f"trend_{pid}")
    trend_chart = create_sales_trend_analysis(forecast_data, trend_key)
    if trend_chart:
        st.plotly_chart(trend_chart, use_container_width=True, key=trend_key)

def display_product_card_shop(pid, details, app):
    """Display product card for shop page"""
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        st.write(f"**{details['name']}**")
        st.write(f"Category: {details['category']}")
        st.write(f"Price: **${details['price']:.2f}**")
    
    with col2:
        current_stock = details['current_stock']
        st.write(f"Available Stock: **{current_stock}**")
        
        # Only show low stock or out of stock messages
        if current_stock <= 0:
            st.error("❌ Out of Stock")
        elif current_stock <= LOW_STOCK_THRESHOLD:
            st.warning("⚠️ Low Stock")
    
    with col3:
        # Buy product section
        if current_stock > 0:
            quantity = st.number_input(
                f"Quantity to buy",
                min_value=1,
                max_value=current_stock,
                value=1,
                key=f"buy_qty_{pid}"
            )
            if st.button(f"🛒 Buy Now", key=f"buy_{pid}"):
                if app.buy_product(pid, quantity):
                    st.success(f"✅ Purchased {quantity} {details['name']}!")
                    st.rerun()
                else:
                    st.error("❌ Purchase failed!")
        else:
            st.info("Out of stock")
    
    st.markdown("---")

def display_low_stock_product(pid, details, app):
    """Display low stock product with simple restock input"""
    col1, col2, col3 = st.columns([3, 1, 2])
    
    with col1:
        st.write(f"**{details['name']}**")
        st.write(f"ID: `{pid}` | Current Stock: **{details['current_stock']}**")
    
    with col2:
        st.error("LOW STOCK")
    
    with col3:
        # Simple restock input
        restock_amount = st.number_input(
            "Restock amount",
            min_value=1,
            max_value=10000,
            value=25,
            key=f"restock_{pid}"
        )
        
        if st.button(f"🔄 Restock", key=f"btn_{pid}"):
            if app.permanent_restock(pid, restock_amount, "Low Stock Restock"):
                st.success(f"✅ Restocked {restock_amount} units!")
                st.rerun()
    
    st.markdown("---")

def show_initial_state(app, products):
    """Show initial state before any sales"""
    st.subheader("📦 Low Stock Products")
    st.info("🤖 AI predictions will appear after products are purchased in the shop.")
    
    # Get low stock products with pagination
    low_stock_products = {pid: details for pid, details in products.items() 
                         if details['current_stock'] <= RESTOCK_THRESHOLD}
    
    if low_stock_products:
        low_stock_list = list(low_stock_products.items())
        total_low_pages = max(1, (len(low_stock_list) + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE)
        
        # Current page low stock products
        start_idx = st.session_state.admin_page * PRODUCTS_PER_PAGE
        end_idx = min(start_idx + PRODUCTS_PER_PAGE, len(low_stock_list))
        current_low_products = low_stock_list[start_idx:end_idx]
        
        st.warning(f"⚠️ Showing {len(current_low_products)} low stock products (Page {st.session_state.admin_page + 1}/{total_low_pages})")
        
        for pid, details in current_low_products:
            display_low_stock_product(pid, details, app)
        
        # Pagination for low stock products
        if total_low_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.session_state.admin_page > 0:
                    if st.button("⬅️ Previous", key="low_prev"):
                        st.session_state.admin_page -= 1
                        st.rerun()
            with col2:
                st.write(f"Page {st.session_state.admin_page + 1} of {total_low_pages}")
            with col3:
                if st.session_state.admin_page < total_low_pages - 1:
                    if st.button("Next ➡️", key="low_next"):
                        st.session_state.admin_page += 1
                        st.rerun()
    else:
        st.success("✅ All products have sufficient stock levels!")

def show_all_purchases_detailed(app):
    """Show ALL purchases with comprehensive information"""
    if not st.session_state.sales_data:
        st.info("No sales data available yet. Make some purchases in the shop!")
        return
    
    st.subheader("💰 ALL Purchase History - Complete Sales Records")
    
    # Create a more detailed sales dataframe
    sales_df = pd.DataFrame(st.session_state.sales_data)
    
    # Sort by timestamp (most recent first)
    sales_df = sales_df.sort_values('timestamp', ascending=False)
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = len(sales_df)
        st.metric("Total Transactions", total_sales)
    
    with col2:
        total_quantity = sales_df['quantity'].sum()
        st.metric("Total Units Sold", total_quantity)
    
    with col3:
        total_revenue = sales_df['total'].sum()
        st.metric("Total Revenue", f"${total_revenue:.2f}")
    
    with col4:
        unique_products = sales_df['product_id'].nunique()
        st.metric("Unique Products Sold", unique_products)
    
    # Show ALL sales transactions with pagination
    st.subheader("📋 Complete Sales Transactions")
    
    # Format the dataframe for better display
    display_df = sales_df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
    display_df['total'] = display_df['total'].apply(lambda x: f"${x:.2f}")
    
    # Rename columns for better display
    display_df = display_df.rename(columns={
        'timestamp': 'Purchase Time',
        'product_id': 'Product ID',
        'product_name': 'Product Name',
        'quantity': 'Quantity',
        'price': 'Unit Price',
        'total': 'Total Amount'
    })
    
    # Show ALL records with pagination
    records_per_page = 20
    total_pages = max(1, len(display_df) // records_per_page + 1)
    
    if 'sales_page' not in st.session_state:
        st.session_state.sales_page = 0
    
    start_idx = st.session_state.sales_page * records_per_page
    end_idx = min(start_idx + records_per_page, len(display_df))
    
    st.write(f"**Showing records {start_idx + 1}-{end_idx} of {len(display_df)} total purchases**")
    
    # Display current page
    current_page_df = display_df.iloc[start_idx:end_idx]
    st.dataframe(current_page_df, use_container_width=True)
    
    # Pagination controls
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
        
        with col2:
            if st.session_state.sales_page > 0:
                if st.button("⬅️ Previous Page", key="sales_prev"):
                    st.session_state.sales_page -= 1
                    st.rerun()
        
        with col3:
            st.write(f"Page {st.session_state.sales_page + 1}/{total_pages}")
        
        with col4:
            if st.session_state.sales_page < total_pages - 1:
                if st.button("Next Page ➡️", key="sales_next"):
                    st.session_state.sales_page += 1
                    st.rerun()
    
    # Sales analysis by product
    st.subheader("📊 Sales Analysis by Product")
    
    product_sales = sales_df.groupby(['product_id', 'product_name']).agg({
        'quantity': 'sum',
        'total': 'sum',
        'timestamp': 'count'
    }).reset_index()
    
    product_sales = product_sales.rename(columns={
        'product_id': 'Product ID',
        'product_name': 'Product Name',
        'quantity': 'Total Units Sold',
        'total': 'Total Revenue',
        'timestamp': 'Number of Transactions'
    })
    
    product_sales['Total Revenue'] = product_sales['Total Revenue'].apply(lambda x: f"${x:.2f}")
    product_sales = product_sales.sort_values('Total Units Sold', ascending=False)
    
    st.dataframe(product_sales, use_container_width=True)

def show_complete_stock_information(app):
    """Show complete stock information for all products"""
    st.subheader("📦 Complete Stock Inventory")
    
    products = st.session_state.products
    if not products:
        st.info("No products available in inventory.")
        return
    
    # Create stock dataframe
    stock_data = []
    for pid, details in products.items():
        stock_status = "✅ Adequate"
        if details['current_stock'] <= 0:
            stock_status = "❌ Out of Stock"
        elif details['current_stock'] <= RESTOCK_THRESHOLD:
            stock_status = "⚠️ Low Stock"
        
        stock_data.append({
            'Product ID': pid,
            'Product Name': details['name'],
            'Category': details['category'],
            'Current Stock': details['current_stock'],
            'Price': f"${details['price']:.2f}",
            'Stock Status': stock_status,
            'Original Stock': details.get('original_stock', details['current_stock'])
        })
    
    stock_df = pd.DataFrame(stock_data)
    stock_df = stock_df.sort_values('Current Stock', ascending=True)
    
    # Display stock summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_products = len(stock_df)
        st.metric("Total Products", total_products)
    
    with col2:
        out_of_stock = len(stock_df[stock_df['Current Stock'] <= 0])
        st.metric("Out of Stock", out_of_stock)
    
    with col3:
        low_stock = len(stock_df[(stock_df['Current Stock'] > 0) & (stock_df['Current Stock'] <= RESTOCK_THRESHOLD)])
        st.metric("Low Stock", low_stock)
    
    with col4:
        adequate_stock = len(stock_df[stock_df['Current Stock'] > RESTOCK_THRESHOLD])
        st.metric("Adequate Stock", adequate_stock)
    
    # Show complete stock table
    st.dataframe(stock_df, use_container_width=True)
    
    # Stock status distribution
    st.subheader("🎯 Stock Status Distribution")
    
    status_counts = stock_df['Stock Status'].value_counts()
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Stock Status Distribution",
        color=status_counts.index,
        color_discrete_map={
            "✅ Adequate": "#4CAF50",
            "⚠️ Low Stock": "#FFA726", 
            "❌ Out of Stock": "#FF6B6B"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

def show_history_sections(app):
    """Show history sections with ALL purchases and complete stock"""
    # Show ALL purchases
    show_all_purchases_detailed(app)
    
    # Show complete stock information
    show_complete_stock_information(app)
    
    # Restock History
    if st.session_state.restock_history:
        st.subheader("📋 Complete Restock History")
        restock_df = pd.DataFrame(st.session_state.restock_history)
        restock_df = restock_df.sort_values('timestamp', ascending=False)
        
        # Format for display
        display_restock = restock_df.copy()
        display_restock['timestamp'] = pd.to_datetime(display_restock['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        display_restock = display_restock.rename(columns={
            'timestamp': 'Restock Time',
            'product_id': 'Product ID',
            'product_name': 'Product Name',
            'quantity_added': 'Quantity Added',
            'new_stock_level': 'New Stock Level',
            'reason': 'Reason'
        })
        
        st.dataframe(display_restock, use_container_width=True)

def show_shop_page(app):
    """Shop page with pagination"""
    st.title("🛍️ Product Store")
    
    products = st.session_state.products
    if not products:
        st.error("No products available")
        return
    
    product_list = list(products.items())
    total_pages = max(1, (len(product_list) + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE)
    
    start_idx = st.session_state.shop_page * PRODUCTS_PER_PAGE
    end_idx = min(start_idx + PRODUCTS_PER_PAGE, len(product_list))
    current_products = product_list[start_idx:end_idx]
    
    st.write(f"**Showing products {start_idx + 1}-{end_idx} of {len(product_list)}**")
    
    for pid, details in current_products:
        display_product_card_shop(pid, details, app)
    
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
        
        with col2:
            if st.session_state.shop_page > 0:
                if st.button("⬅️ Previous", key="shop_prev"):
                    st.session_state.shop_page -= 1
                    st.rerun()
        
        with col3:
            st.write(f"Page {st.session_state.shop_page + 1}/{total_pages}")
        
        with col4:
            if st.session_state.shop_page < total_pages - 1:
                if st.button("Next ➡️", key="shop_next"):
                    st.session_state.shop_page += 1
                    st.rerun()

def show_admin_page(app):
    """Admin page with complete purchase and stock information"""
    st.title("👨‍💼 Admin Dashboard - Complete Overview")
    
    products = st.session_state.products
    if not products:
        st.error("No products available")
        return
    
    stats = app.get_basic_stats()
    
    # Display Basic Information
    st.subheader("📊 Basic Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", stats['total_products'])
    with col2:
        st.metric("Total Sales", stats['total_sales'])
    with col3:
        st.metric("Total Revenue", f"${stats['total_revenue']:.2f}")
    with col4:
        st.metric("Low Stock Items", stats['low_stock_count'])
    
    # Show complete purchase history and stock information
    show_history_sections(app)
    
    # Show low stock products for quick management
    show_initial_state(app, products)

def show_sales_forecasting_dashboard(app):
    """Sales Forecasting Dashboard - PRODUCTS ON TOP, GRAPHS BELOW"""
    st.title("📈 Sales Forecasting Dashboard")
    st.markdown("### 🤖 AI-Powered 3-Month Sales Predictions")
    
    # Check if there are any sales
    if not st.session_state.sales_data:
        st.info("📊 No sales data available yet. Make some purchases in the shop to see sales forecasts.")
        return
    
    # Generate sales forecasts for ALL products with sales history
    with st.spinner("🔮 Analyzing sales patterns and generating 3-month forecasts..."):
        forecasts = app.generate_sales_forecasts_all()
    
    if not forecasts:
        st.warning("No products with sufficient sales history for forecasting.")
        return
    
    # Overview section
    st.subheader("📊 Sales Forecast Overview")
    create_sales_products_overview(forecasts)
    
    # Risk analysis
    st.subheader("🎯 Risk Analysis")
    col1, col2 = st.columns(2)
    with col1:
        risk_key = app.get_unique_key("risk_analysis")
        risk_chart = create_risk_analysis_chart(forecasts, risk_key)
        if risk_chart:
            st.plotly_chart(risk_chart, use_container_width=True, key=risk_key)
    
    with col2:
        # Sales trend summary
        st.subheader("📈 Sales Insights")
        total_forecast = sum(f['total_3month_forecast'] for f in forecasts.values())
        avg_confidence = np.mean([1 if f['confidence'] == 'High' else 0.5 if f['confidence'] == 'Medium' else 0.2 for f in forecasts.values()])
        
        st.metric("Total 3-Month Forecast", f"{total_forecast:.0f} units")
        st.metric("Average Confidence", f"{avg_confidence*100:.0f}%")
        st.metric("Products Analyzed", len(forecasts))
    
    # Individual product forecasts - Show ALL products with sales history
    st.subheader("🛍️ Products with Sales Forecasts")
    
    # Get ALL products with sales forecasts
    forecast_list = list(forecasts.items())
    
    if not forecast_list:
        st.info("No products with sales forecasts available.")
        return
    
    total_forecast_pages = max(1, (len(forecast_list) + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE)
    
    start_idx = st.session_state.forecast_page * PRODUCTS_PER_PAGE
    end_idx = min(start_idx + PRODUCTS_PER_PAGE, len(forecast_list))
    current_forecasts = forecast_list[start_idx:end_idx]
    
    st.write(f"**Showing {len(current_forecasts)} products with sales forecasts (Page {st.session_state.forecast_page + 1}/{total_forecast_pages})**")
    
    # DISPLAY PRODUCTS FIRST (TOP SECTION)
    for pid, forecast_data in current_forecasts:
        if pid in st.session_state.products:
            # Display product forecast card (which now has products on top and graphs below)
            display_sales_forecast_card(pid, st.session_state.products[pid], forecast_data, app)
    
    # Pagination
    if total_forecast_pages > 1:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.session_state.forecast_page > 0:
                if st.button("⬅️ Previous Page", key="sales_prev"):
                    st.session_state.forecast_page -= 1
                    st.rerun()
        with col2:
            st.write(f"Page {st.session_state.forecast_page + 1} of {total_forecast_pages}")
        with col3:
            if st.session_state.forecast_page < total_forecast_pages - 1:
                if st.button("Next Page ➡️", key="sales_next"):
                    st.session_state.forecast_page += 1
                    st.rerun()

def main():
    app = DemandForecastingApp()
    
    # Add a reset button in sidebar for testing (optional)
    with st.sidebar:
        st.markdown("---")
        st.subheader("Data Management")
        if st.button("🔄 Reset All Data"):
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            st.session_state.clear()
            st.rerun()
        
        if st.button("💾 Force Save Data"):
            app.save_data()
            st.success("Data saved successfully!")
    
    page = st.sidebar.selectbox("Menu", ["🛍️ Shop", "👨‍💼 Admin", "📈 Sales Forecasting"])
    
    if page == "🛍️ Shop":
        show_shop_page(app)
    elif page == "👨‍💼 Admin":
        show_admin_page(app)
    else:
        show_sales_forecasting_dashboard(app)

if __name__ == "__main__":
    main()