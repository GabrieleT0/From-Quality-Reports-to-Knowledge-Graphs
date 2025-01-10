import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain.chains import LLMChain
import time

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
huggin_face_token = os.getenv('HUGGIN_FACE_TOKEN')
gemini_key = os.getenv('GOOGLE_AI')

class PromptLLMS:
    def __init__(self, prompt_template, csv_title, csv_content, ontology_content, kg_example = False):
        self.prompt_template = prompt_template
        self.csv_title = csv_title
        self.csv_content = csv_content
        self.ontology_content = ontology_content
        self.kg_example = kg_example

    def execute_on_gemini(self):
        gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=gemini_key,max_tokens=None, temperature=1)
        chain = self.prompt_template | gemini

        if self.kg_example == False:
            result =  chain.invoke({"csv_title": self.csv_title, "csv_content": self.csv_content, "ontology_content" : self.ontology_content})
        else:
            result =  chain.invoke({"csv_title": self.csv_title, "csv_content": self.csv_content, "ontology_content" : self.ontology_content, "kg_example" : self.kg_example})

        return result.content

    def execute_on_gemini_prompt_chaining(self, kgs_number):
        gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=gemini_key,max_tokens=None, temperature=0.7)
    
        refinement_prompt = PromptTemplate(
            input_variables=["initial_message","kgs_number","csv_file"],
            template=(
                "{initial_message}\n{kgs_number}\n{csv_file}"
            )
        )
        chain = refinement_prompt | gemini

        initial_message = f"""
            Consider the following csv entitled {self.csv_title}:\n
            {self.csv_content}
            Consider the following ontology in ttl format entitled 'dqv.ttl':\n
            {self.ontology_content}
            The CSV pasted before contains the quality data of KGs and for each dimension, the file details its metrics with related measurements.
            To distinguish metrics and dimension , consider that all the file column names follow the pattern
            of DIMENSION_METRIC . All the column names ending with -score
            represent the score attached to the dimension reported as prefix of
            the column name. Score and normalized score are two columns representing the overall quality score of KG.
            With this premises, can you model the {self.csv_title} file content according to the dqv.ttl ontology and return the resulting triples in ttl format? 
            Give me the solution for the first KG in the CSV, not a pattern to follow and return me only ttl code, don't add more.
        """
        outputs = []

        for i in range(kgs_number):
            if i == 0:
                current_response = chain.invoke({
                    "initial_message": initial_message,
                    "kgs_number": '',
                    "csv_file" : ''
                })
            else:
                current_response = chain.invoke({
                    "initial_message": previous_answer,
                    "kgs_number": f'Based on the previously response, do the same also for the KG {i + 1} in the csv file:\n',
                    "csv_file": self.csv_content
                })
            previous_answer = current_response.content
            outputs.append(current_response.content)

            print(f"Iteration {i + 1}")
            time.sleep(10)
        
        print(outputs)
        return outputs
    

    def execute_gemini_oneshot_chaining_prompt(self,kgs_number):
        gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=gemini_key,max_tokens=None, temperature=0.7)
    
        refinement_prompt = PromptTemplate(
            input_variables=["initial_message","kgs_number","csv_file"],
            template=(
                "{initial_message}\n{kgs_number}\n{csv_file}"
            )
        )
        chain = refinement_prompt | gemini

        initial_message = f"""
            Consider the following csv entitled {self.csv_title}:\n
            {self.csv_content}
            Consider the following ontology in ttl format entitled 'dqv.ttl':\n
            {self.ontology_content}
            The CSV pasted before contains the quality data of KGs and for each dimension, the file details its metrics with related measurements.
            To distinguish metrics and dimension , consider that all the file column names follow the pattern
            of DIMENSION_METRIC . All the column names ending with -score
            represent the score attached to the dimension reported as prefix of
            the column name. Score and normalized score are two columns representing the overall quality score of KG.
            With this premises, can you model the {self.csv_title} file content according to the dqv.ttl ontology and return the resulting triples in ttl format? 
            Below I show you a complete example modeled for cz-nace, replicate the solution for the first KG in the CSV file, not a pattern to follow and return me only ttl code, don't add more.\n
            Example:\n
            {self.kg_example}
        """
        outputs = []

        for i in range(kgs_number):
            if i == 0:
                current_response = chain.invoke({
                    "initial_message": initial_message,
                    "kgs_number": '',
                    "csv_file" : ''
                })
            else:
                current_response = chain.invoke({
                    "initial_message": previous_answer,
                    "kgs_number": f'Based on the previously response, do the same also for the KG {i + 1} in the csv file:\n',
                    "csv_file": self.csv_content
                })
            previous_answer = current_response.content
            outputs.append(current_response.content)

            print(f"Iteration {i + 1}")
            time.sleep(10)
        
        print(outputs)
        
        return outputs

    def execute_on_openAI_prompt_chaining(self, model_to_use,kgs_number):
        gpt = ChatOpenAI(model=model_to_use,openai_api_key=openai_api_key)
    
        refinement_prompt = PromptTemplate(
            input_variables=["initial_message","kgs_number","csv_file"],
            template=(
                "{initial_message}\n{kgs_number}\n{csv_file}"
            )
        )
        chain = refinement_prompt | gpt

        initial_message = f"""
            Consider the following csv entitled {self.csv_title}:\n
            {self.csv_content}
            Consider the following ontology in ttl format entitled 'dqv.ttl':\n
            {self.ontology_content}
            The CSV pasted before contains the quality data of KGs and for each dimension, the file details its metrics with related measurements.
            To distinguish metrics and dimension , consider that all the file column names follow the pattern
            of DIMENSION_METRIC . All the column names ending with -score
            represent the score attached to the dimension reported as prefix of
            the column name. Score and normalized score are two columns representing the overall quality score of KG.
            With this premises, can you model the {self.csv_title} file content according to the dqv.ttl ontology and return the resulting triples in ttl format? 
            Give me the solution for the first KG in the CSV, not a pattern to follow and return me only ttl code, don't add more.
        """
        outputs = []

        for i in range(kgs_number):
            if i == 0:
                current_response = chain.invoke({
                    "initial_message": initial_message,
                    "kgs_number": '',
                    "csv_file" : ''
                })
            else:
                current_response = chain.invoke({
                    "initial_message": previous_answer,
                    "kgs_number": f'Based on the previously response, do the same also for the KG {i + 1} in the csv file:\n',
                    "csv_file": self.csv_content
                })
            previous_answer = current_response.content
            outputs.append(current_response.content)
            print(f"Iteration {i + 1}")
        
        print(outputs)
        return outputs
    
    def execute_openAI_oneshot_chaining_prompt(self,kgs_number,model_to_use):
        gpt = ChatOpenAI(model=model_to_use,openai_api_key=openai_api_key)
    
        refinement_prompt = PromptTemplate(
            input_variables=["initial_message","kgs_number","csv_file"],
            template=(
                "{initial_message}\n{kgs_number}\n{csv_file}"
            )
        )
        chain = refinement_prompt | gpt

        initial_message = f"""
            Consider the following csv entitled {self.csv_title}:\n
            {self.csv_content}
            Consider the following ontology in ttl format entitled 'dqv.ttl':\n
            {self.ontology_content}
            The CSV pasted before contains the quality data of KGs and for each dimension, the file details its metrics with related measurements.
            To distinguish metrics and dimension , consider that all the file column names follow the pattern
            of DIMENSION_METRIC . All the column names ending with -score
            represent the score attached to the dimension reported as prefix of
            the column name. Score and normalized score are two columns representing the overall quality score of KG.
            With this premises, can you model the {self.csv_title} file content according to the dqv.ttl ontology and return the resulting triples in ttl format? 
            Below I show you a complete example modeled for cz-nace, replicate the solution for the first KG in the CSV file, not a pattern to follow and return me only ttl code, don't add more.\n
            Example:\n
            {self.kg_example}
        """
        outputs = []

        for i in range(kgs_number):
            if i == 0:
                current_response = chain.invoke({
                    "initial_message": initial_message,
                    "kgs_number": '',
                    "csv_file" : ''
                })
            else:
                current_response = chain.invoke({
                    "initial_message": previous_answer,
                    "kgs_number": f'Based on the previously response, do the same also for the KG {i + 1} in the csv file:\n',
                    "csv_file": self.csv_content
                })
            previous_answer = current_response.content
            outputs.append(current_response.content)

            print(f"Iteration {i + 1}")
        
        print(outputs)
        
        return outputs

    def execute_on_gpt_4(openAI_model):
        gpt_4 = ChatOpenAI(model=openAI_model,openai_api_key=openai_api_key,temperature=0.7)
        chain = self.prompt_template | gpt_4

        if self.kg_example == False:
            result =  chain.invoke({"csv_title": self.csv_title, "csv_content": self.csv_content, "ontology_content" : self.ontology_content})
        else:
            result =  chain.invoke({"csv_title": self.csv_title, "csv_content": self.csv_content, "ontology_content" : self.ontology_content, "kg_example" : self.kg_example})

        return result.content

    def execute_on_ollama(self,model_name,api_url):
        ollama = ChatOllama(model=model_name,base_url=api_url,memory=ConversationBufferMemory())
        chain = self.prompt_template | ollama
        
        if self.kg_example == False:
            result =  chain.invoke({"csv_title": self.csv_title, "csv_content": self.csv_content, "ontology_content" : self.ontology_content})
        else:
            result =  chain.invoke({"csv_title": self.csv_title, "csv_content": self.csv_content, "ontology_content" : self.ontology_content, "kg_example" : self.kg_example})

        return result