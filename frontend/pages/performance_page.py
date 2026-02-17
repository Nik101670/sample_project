import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def show_performance_page(API_URL):
    """Display the model performance dashboard"""
    st.title("📊 Model Performance Dashboard")
    
    # Model Information Section
    st.header("🤖 Model Information")
    _display_model_metrics(API_URL)
    
    # Actual vs Predicted Section
    st.header("🎯 Actual vs Predicted Analysis")
    _display_prediction_analysis(API_URL)
    
    # Feature Importance Section
    st.header("🔍 Feature Importance Analysis")
    _display_feature_importance(API_URL)
    
    # Model Training History
    st.header("📈 Training Performance")
    _display_training_history(API_URL)


def _display_model_metrics(API_URL):
    """Display model performance metrics"""
    try:
        model_info_res = requests.get(f"{API_URL}/model-info")
        if model_info_res.status_code == 200:
            model_info = model_info_res.json()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "R² Score", 
                    f"{model_info.get('r2', 0.905):.1%}", 
                    help="Coefficient of determination - higher is better"
                )
            with col2:
                st.metric(
                    "RMSE", 
                    f"{model_info.get('rmse', 3.2):.2f}", 
                    help="Root Mean Square Error in parts/hour"
                )
            with col3:
                st.metric(
                    "MAE", 
                    f"{model_info.get('mae', 2.1):.2f}", 
                    help="Mean Absolute Error in parts/hour"
                )
            with col4:
                st.metric(
                    "Accuracy", 
                    f"{100 - model_info.get('mape', 8.5):.1f}%", 
                    help="Overall prediction accuracy"
                )
                
            # Additional model info
            with st.expander("📋 Detailed Model Information"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.info(f"**Model Version:** {model_info.get('version', '2.0')}")
                    st.info(f"**Features Count:** {model_info.get('features_count', 'N/A')}")
                with col_b:
                    st.info(f"**Training Samples:** {model_info.get('training_samples', 'N/A')}")
                    st.info(f"**Test Samples:** {model_info.get('test_samples', 'N/A')}")
        else:
            st.error("Unable to fetch model information")
    except Exception as e:
        st.error(f"Model info endpoint not available: {e}")


def _display_prediction_analysis(API_URL):
    """Display actual vs predicted analysis with interactive charts"""
    try:
        test_data_res = requests.get(f"{API_URL}/test-predictions")
        if test_data_res.status_code == 200:
            test_data = test_data_res.json()
            df_test = pd.DataFrame(test_data)
            
            # Interactive Scatter Plot
            fig_scatter = px.scatter(
                df_test, 
                x='actual', 
                y='predicted',
                title='Actual vs Predicted Production Output',
                labels={
                    'actual': 'Actual Output (parts/hour)', 
                    'predicted': 'Predicted Output (parts/hour)'
                },
                hover_data=['error', 'error_percentage']
            )
            
            # Add perfect prediction line
            min_val = min(df_test['actual'].min(), df_test['predicted'].min())
            max_val = max(df_test['actual'].max(), df_test['predicted'].max())
            fig_scatter.add_shape(
                type="line",
                x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            fig_scatter.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Error Distribution Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_error_hist = px.histogram(
                    df_test, 
                    x='error',
                    title='Prediction Error Distribution',
                    labels={'error': 'Prediction Error (parts/hour)', 'count': 'Frequency'},
                    nbins=20
                )
                fig_error_hist.update_layout(height=400)
                st.plotly_chart(fig_error_hist, use_container_width=True)
            
            with col2:
                fig_error_box = px.box(
                    df_test,
                    y='error_percentage',
                    title='Prediction Error Percentage',
                    labels={'error_percentage': 'Error Percentage (%)'}
                )
                fig_error_box.update_layout(height=400)
                st.plotly_chart(fig_error_box, use_container_width=True)
            
            # Summary statistics
            with st.expander("📊 Error Statistics"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Mean Error", f"{df_test['error'].mean():.2f}")
                    st.metric("Std Error", f"{df_test['error'].std():.2f}")
                with col_b:
                    st.metric("Mean Abs Error", f"{df_test['error'].abs().mean():.2f}")
                    st.metric("Max Error", f"{df_test['error'].abs().max():.2f}")
                with col_c:
                    accuracy_within_5pct = (df_test['error_percentage'].abs() <= 5).mean() * 100
                    accuracy_within_10pct = (df_test['error_percentage'].abs() <= 10).mean() * 100
                    st.metric("Within 5%", f"{accuracy_within_5pct:.1f}%")
                    st.metric("Within 10%", f"{accuracy_within_10pct:.1f}%")
            
        else:
            st.error("Unable to fetch test predictions")
    except Exception as e:
        st.error(f"Test predictions endpoint not available: {e}")


def _display_feature_importance(API_URL):
    """Display feature importance analysis"""
    try:
        importance_res = requests.get(f"{API_URL}/feature-importance")
        if importance_res.status_code == 200:
            importance_data = importance_res.json()
            
            # Create DataFrame for plotting
            df_importance = pd.DataFrame([
                {
                    'feature': k, 
                    'importance': v, 
                    'direction': 'Positive' if v > 0 else 'Negative',
                    'abs_importance': abs(v)
                }
                for k, v in importance_data.items()
            ])
            df_importance = df_importance.sort_values('abs_importance', ascending=False)
            
            # Top 15 features
            df_top = df_importance.head(15)
            
            fig_importance = px.bar(
                df_top,
                x='importance',
                y='feature',
                color='direction',
                orientation='h',
                title='Top 15 Feature Importance (Model Weights)',
                labels={'importance': 'Feature Weight', 'feature': 'Features'},
                color_discrete_map={'Positive': 'green', 'Negative': 'red'}
            )
            
            fig_importance.update_layout(
                height=600, 
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Feature importance summary
            with st.expander("📋 Feature Insights"):
                positive_features = df_importance[df_importance['direction'] == 'Positive'].head(5)
                negative_features = df_importance[df_importance['direction'] == 'Negative'].head(5)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.success("**🔼 Top Positive Impact Features:**")
                    for _, row in positive_features.iterrows():
                        st.write(f"• {row['feature']}: +{row['importance']:.3f}")
                
                with col_b:
                    st.error("**🔽 Top Negative Impact Features:**")
                    for _, row in negative_features.iterrows():
                        st.write(f"• {row['feature']}: {row['importance']:.3f}")
            
        else:
            st.error("Unable to fetch feature importance")
    except Exception as e:
        st.error(f"Feature importance endpoint not available: {e}")


def _display_training_history(API_URL):
    """Display model training history"""
    try:
        training_res = requests.get(f"{API_URL}/training-history")
        if training_res.status_code == 200:
            training_data = training_res.json()
            
            if 'losses' in training_data:
                df_training = pd.DataFrame({
                    'epoch': range(1, len(training_data['losses']) + 1),
                    'loss': training_data['losses']
                })
                
                fig_training = px.line(
                    df_training,
                    x='epoch',
                    y='loss',
                    title='Model Training Loss Over Time',
                    labels={'epoch': 'Training Epoch', 'loss': 'MSE Loss'}
                )
                
                fig_training.update_layout(height=400)
                st.plotly_chart(fig_training, use_container_width=True)
                
                # Training summary
                with st.expander("🏋️ Training Summary"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Epochs", training_data.get('epochs', 'N/A'))
                    with col_b:
                        st.metric("Final Loss", f"{training_data.get('final_loss', 0):.4f}")
                    with col_c:
                        improvement = (training_data['losses'][0] - training_data['losses'][-1]) / training_data['losses'][0] * 100
                        st.metric("Loss Improvement", f"{improvement:.1f}%")
        else:
            st.info("Training history not available")
    except Exception as e:
        st.info(f"Training history endpoint not available: {e}")