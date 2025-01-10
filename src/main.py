import argparse
import run_on_llms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script with parameter -g o --gemini to run the experiment on Gemini 1.5 pro or -o or --openai to run the experiment on OpenAI models")
    group = parser.add_mutually_exclusive_group()

    group.add_argument("-g", "--gemini", action="store_true", help="If specified, the experiment will be run on Gemini 1.5 pro.")
    group.add_argument("-o", "--openai", action="store_true", help="If specified, the experiment will be run on OpenAI models.")
    group.add_argument("-c", "--claude", action="store_true", help="If specified, the experiment will be run on Claude 3.5 Sonnet.")
    
    args = parser.parse_args()

    # Set the OpenAI model to use
    openAI_model = 'gpt-4o-2024-08-06'

    if(args.gemini):
        run_on_llms.run_experiment_on_gemini()
    if(args.openai):
        run_on_llms.run_experiment_on_openAI_models(openAI_model)
    if(args.claude):
        run_on_llms.run_experiment_on_claude()
    else:
        run_on_llms.run_experiment_on_gemini() 
        run_on_llms.run_experiment_on_openAI_models(openAI_model)
        run_on_llms.run_experiment_on_claude()