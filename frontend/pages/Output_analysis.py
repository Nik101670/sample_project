import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
from scipy import stats


def load_manufacturing_data():
    """Load the manufacturing dataset"""
    try:
        # Try multiple possible locations for the CSV file
        possible_paths = [
            # Current directory (pages/)
            os.path.join(os.path.dirname(__file__), 'manufacturing_dataset_1000_samples.csv'),
            # Model directory (relative to pages/)
            os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'manufacturing_dataset_1000_samples.csv'),
            # Root directory (relative to pages/)
            os.path.join(os.path.dirname(__file__), '..', '..', 'manufacturing_dataset_1000_samples.csv'),
        ]
        
        for csv_path in possible_paths:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                st.success(f"Dataset loaded successfully: {len(df)} samples")
                return df
        
        # If no file found, create sample data to prevent 503 error
        st.warning("📄 CSV file not found. Creating sample data for demonstration.")
        return create_sample_data()
        
    except Exception as e:
        st.error(f" Error loading dataset: {e}")
        st.info(" Creating sample data to prevent service interruption...")
        return create_sample_data()


def create_sample_data():
    """Create sample manufacturing data if CSV is not available"""
    try:
        np.random.seed(42)  # For reproducible results
        n_samples = 100  # Smaller dataset for faster loading
        
        data = {
            'Parts_Per_Hour': np.random.normal(50, 15, n_samples),
            'Efficiency_Score': np.random.normal(75, 20, n_samples),
            'Cooling_Time': np.random.normal(30, 8, n_samples),
            'Injection_Temperature': np.random.normal(200, 25, n_samples),
            'Injection_Pressure': np.random.normal(1500, 200, n_samples),
            'Operator_Experience': np.random.randint(1, 21, n_samples),
            'Total_Cycle_Time': np.random.normal(45, 10, n_samples),
            'Machine_Age': np.random.randint(1, 16, n_samples),
            'Machine_Type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples),
            'Shift': np.random.choice(['Morning', 'Evening', 'Night'], n_samples)
        }
        
        df = pd.DataFrame(data)
        # Ensure positive values for parts per hour
        df['Parts_Per_Hour'] = np.abs(df['Parts_Per_Hour'])
        
        return df
    except Exception as e:
        st.error(f"Failed to create sample data: {e}")
        return None


def create_feature_analysis_chart(df, feature_name, output_col='Parts_Per_Hour', title_prefix=""):
    """Create an analysis chart for a specific feature vs output"""
    
    if feature_name not in df.columns:
        st.warning(f"Feature '{feature_name}' not found in dataset")
        return None
    
    try:
        # Limit data size for performance (sample if too large)
        if len(df) > 500:
            df_sample = df.sample(n=500, random_state=42)
        else:
            df_sample = df.copy()
        
        # Clean data - remove NaN and infinite values
        clean_df = df_sample[[feature_name, output_col]].dropna()
        
        # Handle numerical vs categorical features
        try:
            if df_sample[feature_name].dtype in ['object', 'category'] or df_sample[feature_name].dtype.name == 'string':
                clean_df = clean_df[np.isfinite(clean_df[output_col])]
            else:
                clean_df = clean_df[np.isfinite(clean_df[feature_name]) & np.isfinite(clean_df[output_col])]
        except (TypeError, ValueError):
            clean_df = clean_df[np.isfinite(clean_df[output_col])]
        
        if len(clean_df) < 2:
            st.warning(f"Not enough valid data for {feature_name}")
            return None
        
        # Calculate correlation with timeout protection
        try:
            correlation = clean_df[feature_name].corr(clean_df[output_col])
            if pd.isna(correlation):
                correlation = 0.0
        except Exception:
            correlation = 0.0
        
        # Create scatter plot with simplified settings
        fig = px.scatter(
            clean_df, 
            x=feature_name, 
            y=output_col,
            title=f"{title_prefix}{feature_name.replace('_', ' ').title()}<br><sub>Correlation: {correlation:.3f}</sub>",
            labels={
                feature_name: feature_name.replace('_', ' ').title(),
                output_col: 'Parts Per Hour'
            },
            opacity=0.6,
            height=350  # Fixed height for consistency
        )
        
        # Add trend line only for numerical features with sufficient variation
        try:
            if (df_sample[feature_name].dtype not in ['object', 'category'] and 
                df_sample[feature_name].dtype.name != 'string' and
                len(clean_df[feature_name].unique()) > 1 and
                len(clean_df) >= 10):  # Minimum points for trend line
                
                z = np.polyfit(clean_df[feature_name], clean_df[output_col], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(clean_df[feature_name].min(), clean_df[feature_name].max(), 50)  # Reduced points
                
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=False
                ))
        except Exception:
            pass  # Skip trend line if any issues
        
        # Optimize layout for performance
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showline=True, linecolor='gray', linewidth=1),
            yaxis=dict(showline=True, linecolor='gray', linewidth=1),
            margin=dict(t=60, b=40, l=40, r=40)  # Compact margins
        )
        
        return fig
        
    except Exception as e:
        st.error(f" Error creating chart for {feature_name}: {str(e)[:100]}...")
        return None


def create_categorical_analysis_chart(df, feature_name, output_col='Parts_Per_Hour'):
    """Create box plot for categorical features"""
    
    if feature_name not in df.columns:
        st.warning(f"Feature '{feature_name}' not found in dataset")
        return None
    
    try:
        # Limit data size for performance
        if len(df) > 500:
            df_sample = df.sample(n=500, random_state=42)
        else:
            df_sample = df.copy()
        
        # Clean data
        clean_df = df_sample[[feature_name, output_col]].dropna()
        clean_df = clean_df[np.isfinite(clean_df[output_col])]
        
        if len(clean_df) < 2:
            st.warning(f"Not enough valid data for {feature_name}")
            return None
        
        # Create box plot
        fig = px.box(
            clean_df,
            x=feature_name,
            y=output_col,
            title=f"{feature_name.replace('_', ' ').title()}<br><sub>Impact on Production Output</sub>",
            labels={
                feature_name: feature_name.replace('_', ' ').title(),
                output_col: 'Production Output (parts/hour)'
            },
            color=feature_name,
            height=350
        )
        
        # Add mean points with error handling
        try:
            means = clean_df.groupby(feature_name)[output_col].mean()
            for i, (category, mean_val) in enumerate(means.items()):
                fig.add_trace(go.Scatter(
                    x=[category],
                    y=[mean_val],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='diamond'),
                    name=f'Mean: {mean_val:.1f}',
                    showlegend=False
                ))
        except Exception:
            pass  # Skip means if any issues
        
        fig.update_layout(
            showlegend=False,
            margin=dict(t=60, b=40, l=40, r=40)
        )
        
        return fig
        
    except Exception as e:
        st.error(f" Error creating categorical chart for {feature_name}: {str(e)[:100]}...")
        return None


def show_feature_insights(df, feature_name, output_col='Parts_Per_Hour'):
    """Show statistical insights for a feature"""
    
    # Clean data for correlation calculation
    clean_df = df[[feature_name, output_col]].dropna()
    
    # For numerical features, also check for finite values
    # For categorical features, just use the dropna result
    try:
        if df[feature_name].dtype in ['object', 'category'] or df[feature_name].dtype.name == 'string':
            # Categorical feature - only remove NaN values
            clean_df = clean_df[np.isfinite(clean_df[output_col])]
        else:
            # Numerical feature - remove NaN and infinite values from both
            clean_df = clean_df[np.isfinite(clean_df[feature_name]) & np.isfinite(clean_df[output_col])]
    except (TypeError, ValueError):
        # If there's any issue with finite check, just use dropna result
        clean_df = clean_df[np.isfinite(clean_df[output_col])]
    
    if len(clean_df) < 2:
        st.warning("Not enough valid data for insights")
        return
    
    # Calculate correlation for numerical features
    is_categorical = df[feature_name].dtype in ['object', 'category'] or df[feature_name].dtype.name == 'string'
    
    if not is_categorical:
        try:
            correlation = clean_df[feature_name].corr(clean_df[output_col])
            if pd.isna(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        # Show impact using emoji and correlation strength
        if abs(correlation) > 0.3:
            strength = "Strong"
        elif abs(correlation) > 0.1:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        if correlation > 0:
            st.markdown(f'✅ **Positive Impact**: Higher values tend to increase production output')
        elif correlation < 0:
            st.markdown(f'❌ **Negative Impact**: Higher values tend to decrease production output')
        else:
            st.markdown(f'➖ **Neutral Impact**: No clear relationship with production output')
    
    # Show optimal range/category (for all features, not just high correlation)
    try:
        # Get top quartile of output for optimal recommendations
        if len(clean_df) >= 4:  # Need at least 4 samples for quartiles
            top_quartile_threshold = clean_df[output_col].quantile(0.75)
            top_quartile = clean_df[clean_df[output_col] >= top_quartile_threshold]
            
            if len(top_quartile) > 0:
                if is_categorical:
                    # For categorical features, show most common category in high output samples
                    most_common = top_quartile[feature_name].mode()
                    if len(most_common) > 0:
                        frequency = (top_quartile[feature_name] == most_common.iloc[0]).sum()
                        percentage = (frequency / len(top_quartile)) * 100
                        st.info(f"**Optimal Category**: {most_common.iloc[0]} ({percentage:.1f}% of high performers)")
                else:
                    # For numerical features, show range where high output occurs
                    optimal_min = top_quartile[feature_name].min()
                    optimal_max = top_quartile[feature_name].max()
                    optimal_mean = top_quartile[feature_name].mean()
                    
                    if all(pd.notna([optimal_min, optimal_max, optimal_mean])):
                        st.info(f"**Optimal Range**: {optimal_min:.2f} - {optimal_max:.2f} (Mean: {optimal_mean:.2f})")
        else:
            st.warning("Insufficient data for optimal range calculation")
            
    except Exception as e:
        st.warning(f"Could not calculate optimal range: {str(e)}")


# Main Analytics Dashboard
st.set_page_config(page_title="Manufacturing Analytics", layout="wide")

st.title(" Feature Impact Analysis")
st.markdown("###  Key Manufacturing Features vs Production Output")

# Add loading state
with st.spinner(' Loading manufacturing data...'):
    df = load_manufacturing_data()

if df is not None:
    # Define the 9 most important features based on your analysis
    top_features = [
        'Efficiency_Score',      # Positive effect
        'Cooling_Time',          # Positive effect
        'Injection_Temperature', # Positive effect
        'Injection_Pressure',    # Positive effect
        'Operator_Experience',   # Positive effect
        'Total_Cycle_Time',      # Negative effect
        'Machine_Age',           # Negative effect
        'Machine_Type',          # Negative effect (categorical)
        'Shift'                  # Negative effect (categorical)
    ]
    
    # Define categorical features
    categorical_features = ['Machine_Type', 'Shift']
    
    # Filter features that exist in the dataset
    available_features = [f for f in top_features if f in df.columns]
    
    if not available_features:
        st.error("None of the important features found in the dataset. Please check feature names.")
        st.write("Available columns:", df.columns.tolist())
    else:
        # Summary statistics
        st.header(" Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Avg Production Output", f"{df['Parts_Per_Hour'].mean():.1f}")
        with col3:
            st.metric("Output Range", f"{df['Parts_Per_Hour'].min():.1f} - {df['Parts_Per_Hour'].max():.1f}")
        with col4:
            st.metric("Features Analyzed", len(available_features))
        
        st.divider()
        
        # Create visualizations for each feature
        st.header(" Individual Feature Analysis")
        
        # Add feature selection to reduce server load
        selected_features = st.multiselect(
            "Select features to analyze (choose fewer for faster loading):",
            available_features,
            default=available_features[:4],  # Default to first 4 to reduce load
            help=" Tip: Select fewer features for faster loading on slower connections"
        )
        
        if not selected_features:
            st.warning(" Please select at least one feature to analyze")
        else:
            # Progress bar for better UX
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Calculate grid dimensions based on selected features
            n_features = len(selected_features)
            n_cols = min(3, n_features)  # Max 3 columns
            n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
            
            # Create dynamic grid
            for row in range(n_rows):
                cols = st.columns(n_cols)
                
                for col_idx in range(n_cols):
                    feature_idx = row * n_cols + col_idx
                    
                    if feature_idx < len(selected_features):
                        feature = selected_features[feature_idx]
                        
                        # Update progress
                        progress = (feature_idx + 1) / len(selected_features)
                        progress_bar.progress(progress)
                        status_text.text(f" Analyzing {feature.replace('_', ' ')}... ({feature_idx + 1}/{len(selected_features)})")
                        
                        with cols[col_idx]:
                            try:
                                # Check if feature is categorical
                                is_categorical = feature in categorical_features
                                
                                with st.spinner(f'Creating chart for {feature}...'):
                                    if is_categorical:
                                        fig = create_categorical_analysis_chart(df, feature)
                                    else:
                                        fig = create_feature_analysis_chart(df, feature, title_prefix=f"{feature_idx+1}. ")
                                
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                                    show_feature_insights(df, feature)
                                else:
                                    st.error(f" Could not create chart for {feature}")
                                    
                            except Exception as e:
                                st.error(f" Error analyzing {feature}: {str(e)[:100]}...")
                                st.info(" Try refreshing the page or selecting fewer features")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            st.success(f" Analysis complete! {len(selected_features)} features analyzed.")
        
        
        # Feature Importance Summary
        st.divider()
        st.header(" Key Insights")
        
        with st.expander(" Feature Impact Summary"):
            for i, feature in enumerate(available_features, 1):
                try:
                    if feature in categorical_features:
                        # For categorical features, we can't calculate traditional correlation
                        st.write(f"**{i}. {feature.replace('_', ' ').title()}**: Categorical feature - impact shown in box plot above")
                    else:
                        correlation = df[feature].corr(df['Parts_Per_Hour'])
                        impact_type = "Positive" if correlation > 0 else "Negative"
                        impact_strength = "Strong" if abs(correlation) > 0.3 else "Moderate" if abs(correlation) > 0.1 else "Weak"
                        
                        st.write(f"**{i}. {feature.replace('_', ' ').title()}**: {impact_strength} {impact_type} impact (r={correlation:.3f})")
                except:
                    st.write(f"**{i}. {feature.replace('_', ' ').title()}**: Unable to calculate correlation")
        
        with st.expander(" Optimization Recommendations"):
            st.write("**For Maximum Production Output:**")
            
            # Get top quartile performers
            top_performers = df[df['Parts_Per_Hour'] >= df['Parts_Per_Hour'].quantile(0.75)]
            
            for feature in available_features:  # Show all 9 recommendations
                if feature in top_performers.columns:
                    try:
                        if feature in categorical_features:
                            # For categorical features, show most common value in top performers
                            most_common = top_performers[feature].mode()
                            if len(most_common) > 0:
                                st.write(f"• **{feature.replace('_', ' ')}**: Use '{most_common.iloc[0]}'")
                        else:
                            # For numerical features, use correlation-based recommendations
                            correlation = df[feature].corr(df['Parts_Per_Hour'])
                            optimal_mean = top_performers[feature].mean()
                            
                            if abs(correlation) > 0.1:  # Only show meaningful recommendations
                                if correlation > 0:
                                    recommendation = f"Increase {feature.replace('_', ' ')} towards {optimal_mean:.2f}"
                                else:
                                    recommendation = f"Decrease {feature.replace('_', ' ')} towards {optimal_mean:.2f}"
                                
                                st.write(f"• {recommendation}")
                    except:
                        # Skip if there are issues with the feature
                        pass

else:
    st.error(" Could not load the manufacturing dataset.")
    
    with st.expander("🔧 Troubleshooting Information"):
        st.write("**Possible solutions:**")
        st.write("• Check that 'manufacturing_dataset_1000_samples.csv' exists in the model directory")
        st.write("• Verify file permissions and accessibility")
        st.write("• Try refreshing the page")
        st.write("• Contact support if the issue persists")
        
        st.write("**Expected file locations:**")
        st.code("""
        model/manufacturing_dataset_1000_samples.csv
        OR
        frontend/pages/manufacturing_dataset_1000_samples.csv
        """)