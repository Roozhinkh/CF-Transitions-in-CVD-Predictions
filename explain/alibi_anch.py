from alibi.explainers import AnchorTabular
import pandas as pd

ESS_TABLE = """
Core variables according to ESS documentation:
 - etfruit  : frequency of fruit consumption
 - eatveg   : frequency of vegetable consumption
 - cgtsmok  : smoking behavior
 - alcfreq  : alcohol consumption frequency
 - slprl    : sleep problems
 - paccnois : exposure to noise
 - bmi      : body mass index (newly created)
 - gndr     : gender
Additionally included:
 - health   : self-rated health
 - dosprt   : sport/physical activity
 - sclmeet  : frequency of social meetings
 - inprdsc  : perceived discrimination
 - ctrlife  : perceived control over life
"""


def explain_with_alibi_anchors(model, X, query_instance, feature_cols):
    #Generate Anchor explanations for the model's prediction.
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

    print(ESS_TABLE)

    print(f"Anchor rules: \n{'\n'.join(explanation.anchor)} \n")
    print(f"Precision: {explanation.precision:.3f}")
    print(f"Coverage: {explanation.coverage:.3f}")
    print("==================================\n")
    return explanation