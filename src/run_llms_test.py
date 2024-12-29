from langchain_core.prompts import PromptTemplate
from prompt_llms import PromptLLMS
from evaluate_answer import EvaluateKG


csv_path = './Light/only-availability.csv'
ontology_path = './dqv.ttl'
kg_as_example_path = './Full/cz-nace-full.ttl'

# Read and transform the csv file into text
with open(csv_path) as f:
    csv_text = f.read() + '\n'

# Read and trasfromt the ttl ontology into text
with open(ontology_path) as f:
    ttl_text = f.read() + '\n'

# Read and trasform the ttl KG in a string
with open(kg_as_example_path) as f:
    kg_as_example = f.read() + '\n'

#Zero-shot Prompting Complete 
zero_shot_prompt = PromptTemplate(
    input_variables=["csv_title","csv_content","ontology_content"],
    template='''Consider the following csv entitled {csv_title}: {csv_content} \n
                Consider the following ontology in ttl format entitled 'dqv.ttl':{ontology_content} \n
                The CSV pasted before contains the quality data of KGs and for each dimension, the file details its metrics with related measurements. To distinguish metrics and
                dimension , consider that all the file column names follow the pattern
                of DIMENSION_METRIC . All the column names ending with -score
                represent the score attached to the dimension reported as prefix of
                the column name. Score and normalized score are two columns representing the overall quality score of KG.
                With this premises, can you model the {csv_title} file content according to the dqv.ttl ontology and return the resulting triples in rdf format? 
                Return me only ttl code, don't add more \n
    '''
)

#Zero-shot Prompting only Accessibility 
zero_shot_prompt = PromptTemplate(
    input_variables=["csv_title","csv_content","ontology_content"],
    template='''Consider the following csv entitled {csv_title}: {csv_content} \n
                Consider the following ontology in ttl format entitled 'dqv.ttl':{ontology_content} \n
                The CSV pasted before contains the quality data of KGs with scores and values attached to all dimensions concerning the Accessibility category and for each dimension, the file details its metrics with related measurements. 
                To distinguish metrics and dimension , consider that all the file column names follow the pattern
                of DIMENSION_METRIC . All the column names ending with -score
                represent the score attached to the dimension reported as prefix of
                the column name. Score and normalized score are two columns representing the overall quality score of KG.
                With this premises, can you model the {csv_title} file content according to the dqv.ttl ontology and return the resulting triples in rdf format? 
                Return me only ttl code, don't add more \n
    '''
)

#One-shot Prompting only Accessibility pattern
one_shot_prompt = PromptTemplate(
    input_variables=["csv_title","csv_content","ontology_content","kg_example"],
    template='''Consider the following csv entitled "{csv_title}".csv: {csv_content} \n
    Consider the following ontology in ttl format entitled "dqv.ttl": {ontology_content} \n
    The CSV pasted before contains the quality data of KGs with scores and values attached to all dimensions concerning the Accessibility category and for each dimension, the file details its metrics with related measurements. To distinguish metrics and dimension, consider that all the file column names follow the pattern of DIMENSION_METRIC.
    All the column names ending with -score represent the score attached to the dimension reported as prefix of the column name. Score and normalized score are two columns representing the overall quality score of KG. 
    With these premises, can you model the data contained in csv file according to the "dqv.ttl" ontology and return the complete and detailed set of resulting triples in rdf format? Below I show you an example that models only some of the dimensions and metrics from the CSV file, replicate the pattern and complete it, for all the KGs in the CSV file: \n
    {kg_example}
    \n  Return me only ttl code, don't add more.
    '''
)

#One-shot Prompting only Accessibility complete example
one_shot_prompt = PromptTemplate(
    input_variables=["csv_title","csv_content","ontology_content","kg_example"],
    template='''Consider the following csv entitled "{csv_title}".csv: {csv_content} \n
    Consider the following ontology in ttl format entitled "dqv.ttl": {ontology_content} \n
    The CSV pasted before contains the quality data of KGs with scores and values attached to all dimensions concerning the Accessibility category and for each dimension, the file details its metrics with related measurements. To distinguish metrics and dimension, consider that all the file column names follow the pattern of DIMENSION_METRIC.
    All the column names ending with -score represent the score attached to the dimension reported as prefix of the column name. Score and normalized score are two columns representing the overall quality score of KG. 
    With these premises, can you model the data contained in csv file according to the "dqv.ttl" ontology and return the complete and detailed set of resulting triples in rdf format? Below I show you a complete example modeled for cz-nace, replicate the solution for all the KGs you see in the CSV file: \n
    {kg_example}
    \n  Return me only ttl code, don't add more.
    '''
)

#One-shot Prompting Complete with pattern example
one_shot_prompt = PromptTemplate(
    input_variables=["csv_title","csv_content","ontology_content","kg_example"],
    template='''Consider the following csv entitled "{csv_title}".csv: {csv_content} \n
    Consider the following ontology in ttl format entitled "dqv.ttl": {ontology_content} \n
    The CSV pasted before contains the quality data of KGs and for each dimension, the file details its metrics with related measurements. To distinguish metrics and dimension, consider that all the file column names follow the pattern of DIMENSION_METRIC.
    All the column names ending with -score represent the score attached to the dimension reported as prefix of the column name. Score and normalized score are two columns representing the overall quality score of KG. 
    With these premises, can you model the data contained in csv file according to the "dqv.ttl" ontology and return the complete and detailed set of resulting triples in rdf format? Below I show you an example that models only some of the dimensions and metrics from the CSV file, replicate the pattern and complete it, for all the KGs in the CSV file: \n
    {kg_example}
    \n  Return me only ttl code, don't add more.
    '''
)

#One-shot Prompting Complete
one_shot_prompt = PromptTemplate(
    input_variables=["csv_title","csv_content","ontology_content","kg_example"],
    template='''Consider the following csv entitled "{csv_title}".csv: {csv_content} \n
    Consider the following ontology in ttl format entitled "dqv.ttl": {ontology_content} \n
    The CSV pasted before contains the quality data of KGs and for each dimension, the file details its metrics with related measurements. To distinguish metrics and dimension, consider that all the file column names follow the pattern of DIMENSION_METRIC.
    All the column names ending with -score represent the score attached to the dimension reported as prefix of the column name. Score and normalized score are two columns representing the overall quality score of KG. 
    With these premises, can you model the data contained in csv file according to the "dqv.ttl" ontology and return the complete and detailed set of resulting triples in rdf format? Below I show you a complete example modeled for cz-nace, replicate the solution for all the KGs you see in the CSV file: \n
    {kg_example}
    \n  Return me only ttl code, don't add more.
    '''
)


llms = PromptLLMS(zero_shot_prompt,'only_availability',csv_text,ttl_text,kg_as_example_path)
kg_generated_gemini = llms.execute_on_gemini()
kg_generated_gemini = kg_generated_gemini.replace('`','')
print(kg_generated_gemini)

parsed_kg_gemini = EvaluateKG(kg_generated_gemini,'Gemini 1.5 pro')
evaluation_result = parsed_kg_gemini.execute_evaluation(10,1,6)

print(evaluation_result)