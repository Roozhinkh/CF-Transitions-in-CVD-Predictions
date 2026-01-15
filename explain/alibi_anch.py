from alibi.explainers import AnchorTabular
import pandas as pd


# Generate Anchor explanations for the model's prediction.
# Anchor explanations provide interpretable rules that explain the model's decision
# for a specific instance. As long as the anchor conditions are met, the model's prediction
# is expected to remain the same with high precision.
# Detailed documentation: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/Anchors.ipynb

def explain_with_alibi_anchors(model, X, query_instance, feature_cols):
    """
    Generate Anchor explanations for the model's prediction.
    
    Args:
        model: Trained sklearn classifier model that implements predict() and predict_proba() methods.
        X (pd.DataFrame): Training data features used to fit the anchor explainer.
        query_instance (pd.DataFrame): Single row DataFrame containing the instance to explain.
        feature_cols (list): List of feature column names in the dataset.
    
    Returns:
        explanation: Anchor explanation object containing the anchor rules, precision, and coverage.
    """

    print("\nAlibi Anchor Explanations")

    def predictor_wrapper(X_array):
    # Wrapper to ensure predictions use proper feature names.
        if not isinstance(X_array, pd.DataFrame):
            X_df = pd.DataFrame(X_array, columns=feature_cols)
        else:
            X_df = X_array
        return model.predict(X_df)
    
    # Initialize Anchor explainer
    explainer = AnchorTabular(
        predictor=predictor_wrapper,
        feature_names=feature_cols
    )
    
    # Fit on training data
    explainer.fit(X.values, disc_perc=[25, 50, 75])
    
    # Explain the query instance
    explanation = explainer.explain(query_instance.values[0], threshold=0.95)

    print(f"Anchor rules: {' AND '.join(explanation.anchor)}")
    print(f"Precision: {explanation.precision:.3f}")
    print(f"Coverage: {explanation.coverage:.3f}")
    print("==================================\n")
    return explanation