from alibi.explainers import TreeShap
import pandas as pd

def explain_with_alibi_shap(model, X, query_instance, feature_cols, output_path="", filename="alibi_shap_feature_importance.csv"):
    print("\n=== Alibi SHAP Explanations ===")
    
    
    # Initialize Alibi TreeShap explainer with the model directly
    # TreeShap needs access to the tree structure, not just predictions
    explainer = TreeShap(model)
    
    # Use a subset of data to speed up computation (as recommended in the warning)
    X_background = X.sample(n=min(100, len(X)), random_state=42)
    # explainer.fit(X_background.values)
    explainer.fit(X.values)
    

    # Explain the query instance
    explanation = explainer.explain(query_instance.values)  
    print("\nSHAP values for the query instance:")
    shap_values = explanation.shap_values[1][0]  # Assuming binary classification, get SHAP values for positive class   
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'shap_value': shap_values,
        'feature_value': query_instance.values[0]
    }).sort_values('shap_value', key=abs, ascending=False)
    if output_path != "":
        output_path = output_path
        output_path.mkdir(parents=True, exist_ok=True)
        feature_importance.to_csv(output_path / filename, index=False)
    print(feature_importance.head(10))
    print("Alibi SHAP feature importance saved to:", output_path / filename)
    print("==================================\n")
    return explanation

