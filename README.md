# scoring_credit_streamlit

Implementation of a credit scoring Dashboard

Files:

app_dashboard.py: contains the code to request in the API:
			the probility for the client to be classified in risky or not,
			the classification prediction.
				contains the code to explain these predictions for each client 

explainer.bz2: contains the the shap explainer to explain variable influences

shap_values: contains the array of variable influences by client

requirements: contains all library needed in the app code

sample: sample with 10% of the data before applying normalisation

sample_norm: sample normalized to be applied in the model

logo-pad: the logo of the project

col_shap_most_importance_dic: dictionnary with client and his 20 most influent variables

columns_descrition_dic: dictionnary with variables and there descriptions