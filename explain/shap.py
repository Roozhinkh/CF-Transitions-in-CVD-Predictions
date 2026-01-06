import shap
import pandas as pd

def explain_with_shap(model, X, query_instance, feature_cols, output_path="", filename="shap_feature_importance.csv"):
    """Generate SHAP explanations for feature importance."""
    print("\n=== SHAP Feature Importance ===")
    
    # Create SHAP explainer (TreeExplainer for Random Forest)
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for query instance
    shap_values = explainer.shap_values(query_instance)
    
    # Get SHAP values for positive class (CVD risk)
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1][0]
    else:
        shap_values_pos = shap_values[0, :, 1]
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'shap_value': shap_values_pos,
        'feature_value': query_instance.values[0]
    }).sort_values('shap_value', key=abs, ascending=False)
    if output_path != "":
        output_path = output_path
        output_path.mkdir(parents=True, exist_ok=True)
        feature_importance.to_csv(output_path / filename, index=False)
    
    print(feature_importance.head(10))
    
    return shap_values