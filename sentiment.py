from fastapi import FastAPI
import os
from pydantic import BaseModel
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = FastAPI()

os.environ['GOOGLE_API_KEY'] = "AIzaSyAbhkn4KQxk8lBNtVEF3sNXV1e47SzO2Ic"

llm = GooglePalm(temperature = 0.3)


class InputData(BaseModel):
    news: str

@app.post("/predict")
async def generate_text(data: InputData):

    title_template = PromptTemplate(
        input_variables = ['news'],
        template = "You need to assess this news article and assign it a sentiment either booming or controversial, News articles that have a good sentiment and are being praised need to be classified as booming and articles that are being criticized or have a bad sentiment need to be classified as controversial, Asses this news article and give an output with just the classified label. Here is the news article : {news} "
    )

    title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'output')

    response = title_chain({'news' : data.news})

    return response["output"]
