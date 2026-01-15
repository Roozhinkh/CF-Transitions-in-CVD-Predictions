from alibi.explainers import TreeShap
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


# I have implemented one way of visualising SHAP values with shap.summary_plot found most of the code here: https://github.com/ramonpzg/alibi/blob/9714f6455e007ab3dc4b2fba2ce94385014d7727/doc/source/examples/path_dependent_tree_shap_adult_xgb.ipynb
# Documentation for Alibi's TreeShap explainer: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/TreeSHAP.ipynb
def explain_with_alibi_shap(model, X, query_instance, feature_cols, output_path="", filename="alibi_shap_feature_importance.csv"):
    """
    Explain model predictions using SHAP (SHapley Additive exPlanations) values.
    
    Args:
        model: Trained tree-based model (e.g., RandomForest, XGBoost) to explain
        X: pandas DataFrame containing the training/background data used to compute SHAP values
        query_instance: pandas DataFrame (single row) - the specific instance to explain
        feature_cols: list of str - names of features in the same order as model input
        output_path: str or Path - directory to save outputs (CSV and plot). Default "" means no saving
        filename: str - name for the CSV file with feature importance. Default "alibi_shap_feature_importance.csv"
    
    Returns:
        explanation: Alibi explanation object containing SHAP values and metadata
    """
    print("\n=== Alibi SHAP Explanations ===")
    
    
    # Initialize Alibi TreeShap explainer with the model directly
    explainer = TreeShap(model)
    
    # Use a subset of data to speed up computation (as recommended by alibi)
    X_background = X.sample(n=min(100, len(X)), random_state=42)
    explainer.fit(X_background.values)
    

    # Explain the query instance
    explanation = explainer.explain(query_instance.values)  
    print("\nSHAP values for the query instance:")

    # shap_values: a list of length equal to the number of model outputs,
    # where each entry is an array of dimension samples x features of shap values.
    # Assuming binary classification, get SHAP values for positive class   
    shap_values_test = explanation.shap_values[0]
    shap_values = explanation.shap_values[1][0]  


    # Create feature importance dataframe and order by absolute SHAP value
    # negative values indicate a decrease in predicted risk, positive values an increase
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'shap_value': shap_values,
        'feature_value': query_instance.values[0]
    }).sort_values('shap_value', key=abs, ascending=False)

    print(np.array(shap_values_test).shape)
    

    # Save to CSV if output path is provided
    if output_path != "":
        output_path = output_path
        output_path.mkdir(parents=True, exist_ok=True)
        feature_importance.to_csv(output_path / filename, index=False)
        print("Alibi SHAP feature importance saved to:", output_path / filename)

    # Save SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values_test, query_instance, feature_cols, plot_type='bar', show=False)
    if output_path != "":
        plt.savefig(output_path / 'shap_summary_plot.png', bbox_inches='tight', dpi=150)
        print(f"SHAP summary plot saved to: {output_path / 'shap_summary_plot.png'}")
    plt.close()

    # Display the feature importance
    print(feature_importance.head(len(feature_cols)))
    print("==================================\n")

    return explanation

