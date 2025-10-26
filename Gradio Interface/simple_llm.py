from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams 
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM

model_id = "ibm/granite-3-3-8b-instruct"

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5,
}

project_id = "skills-network"

watsonx_llm = WatsonxLLM(
    model_id = model_id,
    url="https://us-south.ml.cloud.ibm.com",
    project_id = project_id,
    params = parameters,
)

query = input("Please enter your query:")

print(watsonx_llm.invoke(query))