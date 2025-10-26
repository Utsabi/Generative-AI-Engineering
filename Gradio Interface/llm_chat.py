from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams 
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM
import gradio as gr

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

def generate_response(prompt_txt):
    generated_response = watsonx_llm.invoke(prompt_txt)
    return generated_response


chat_application = gr.Interface(
    fn=generate_response,
    allow_flagging="manual",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the app
chat_application.launch(server_name="127.0.0.1", server_port= 7860)
