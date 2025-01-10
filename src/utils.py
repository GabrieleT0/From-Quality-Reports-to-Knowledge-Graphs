import os

here = os.path.dirname(os.path.abspath(__file__))

def trasform_file_to_text(csv_path, ontology_path):
    """
    Reads and transforms the contents of a CSV file, an ontology file, and a knowledge graph (KG) example file into text.

    Args:
        csv_path (str): The file path to the CSV file.
        ontology_path (str): The file path to the ontology file in TTL format.
        kg_as_example_path (str): The file path to the knowledge graph example file in TTL format.

    Returns:
        tuple: A tuple containing three strings:
            - csv_text (str): The content of the CSV file as a string.
            - ttl_text (str): The content of the ontology file as a string.
            - kg_as_example (str): The content of the knowledge graph example file as a string.
    """
    # Read and transform the csv file into text
    with open(csv_path) as f:
        csv_text = f.read() + '\n'

    # Read and trasfromt the ttl ontology into text
    with open(ontology_path) as f:
        ttl_text = f.read() + '\n'

    return csv_text, ttl_text

def convert_example_to_txt(kg_as_example_path):
    """
    Reads and converts the contents of a knowledge graph (KG) example file into text.

    Args:
        kg_as_example_path (str): The file path to the knowledge graph example file in TTL format.

    Returns:
        str: The content of the knowledge graph example file as a string.
    """
    # Read and convert the knowledge graph example file into text
    with open(kg_as_example_path) as f:
        kg_as_example = f.read() + '\n'

    return kg_as_example

def convert_response_to_text(lls_response):
    """
    Converts the given LLM response to a text format.
    If the response is a string, it removes certain substrings such as backticks, 'ttl', and 'turtle'.
    If the response is a list, it concatenates all elements into a single string separated by newlines (prompt-chaining case).
    Args:
        lls_response (str or list): The response from the LLM, either as a single string or a list of strings.
    Returns:
        str: The processed response as a single string.
    """
    if isinstance(lls_response,str):
        response = lls_response.replace('`','')
        response = response.replace('ttl','')
        response = response.replace('turtle','')

        return response
    else:
        response = ''
        for el in lls_response:
            response += '\n' +  el
    
        return response

def save_resonse_as_file(output_filename,response):
    """
    Saves the given response as a file.
    Args:
        output_path (str): The file path to save the response to.
        response (str): The response to save as a file.
    """
    # Path to the output file where the LLM response will be saved
    output_file = os.path.join(here,output_filename + '.txt')

    with open(output_file, 'w', encoding="utf-8") as f:
        f.write(response)