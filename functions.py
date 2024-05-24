import openai
import ast
import re
import pandas as pd
import json


def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''
    
    delimiter = "####"
    example_user_req = {'Origin':'Delhi', 'Destination':'Manali', 'Point of interest':'Hadimba Temple', 'Hotel': 'Baragarh Regency', 'Budget': '150000'}
    
    system_message = f"""

    You are an intelligent travel agent and your goal is to create iternary to users.
    You need to ask relevant questions and understand the user profile by analysing the user's responses.
    You final objective is to fill the values for the different keys ('Source','Destination','Duration','Budget') in the python dictionary and be confident of the values.
    These key value pairs define the user's profile.
    The python dictionary looks like this {{'Source': 'values','Destination': 'values','Duration': 'values','Budget': 'values'}}
    The values for keys 'Source' and 'Destination' should be any city in India. The values for 'Duration' should be in number between 2 to 12. The value for 'Budget', should be in number. All the values will come from the user. 
    The values currently in the dictionary are only representative values. 
    
    {delimiter}Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised.
    - The values for keys 'Source' and 'Destination' should be any city in India extracted from user's response.
    - The values for 'Duration' should be in number between 2 to 12  extracted from user's response.
    - The value for 'Budget', should be a numerical value extracted from user's response.
    - 'Budget' value needs to be greater than or equal to 4000 INR. If the user says less than that, please mention that traveling will require a larger budget.
    - Do not randomly assign values to any of the keys. The values need to be inferred from the user's response.
    {delimiter}

    To fill the dictionary, you need to have the following chain of thoughts:
    {delimiter} Thought 1: Ask a question to understand the user's profile and requirements. \n
    You are trying to fill the values of all the keys ('Source','Destination','Duration','Budget') in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys. 
    Answer "Yes" or "No" to indicate if you understand the requirements and have updated the values for the relevant keys. \n
    If yes, proceed to the next step. Otherwise, rephrase the question to capture their profile. \n{delimiter}

    {delimiter}Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step. 
    Remember the instructions around the values for the different keys. Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    Answer "Yes" or "No" to indicate if you understood all the values for the keys and are confident about the same. 
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.{delimiter}

    {delimiter}Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary. 
    If you are not confident about any of the values, ask clarifying questions. {delimiter}

    Follow the above chain of thoughts and only output the final updated python dictionary. \n


    {delimiter} Here is a sample conversation between the user and assistant:
    User: "Hi, I want to plan a trip to Delhi."
    Assistant: "Sure, let me help you with that! Where are you flying from?"
    User: "I will be flying from Mumbai."
    Assistant: "Thank you for providing that information. How many days are you planning on your trip?"
    User: "I am looking at 4 days."
    Assistant: "Thank you for the information. What is your budget for the entire trip?"
    User: "my max budget is 30000 inr"
    Assistant: "{example_user_req}"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements. Do not start with Assistant: "
    """
    conversation = [{"role": "system", "content": system_message}]
    return conversation



def get_chat_model_completions(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
        max_tokens = 300
    )
    return response.choices[0].message["content"]



def moderation_check(user_input):
    response = openai.Moderation.create(input=user_input)
    moderation_output = response["results"][0]
    if moderation_output["flagged"] == True:
        return "Flagged"
    else:
        return "Not Flagged"


    
def intent_confirmation_layer(response_assistant):
    delimiter = "####"
    prompt = f"""
    You are a senior evaluator who has an eye for detail.
    You are provided an input. You need to evaluate if the input has the following keys: 'Source','Destination','Duration','Budget'
    Next you need to evaluate if the keys have the the values filled correctly.
    The values for keys 'Source' and 'Destination' should be any city in India extracted from user's response. The values for 'Duration' should be in number between 2 to 12  extracted from user's response.
    The value for 'Budget', should be a numerical value in currency INR extracted from user's response. Only extract the numerical value from user's response.
    Output a string 'Yes' if the input contains the dictionary with the values correctly filled for all keys.
    Otherwise out the string 'No'.

    Here is the input: {response_assistant}
    Only output a one-word string - Yes/No.
    """


    confirmation = openai.Completion.create(
                                    model="text-davinci-003",
                                    prompt = prompt,
                                    temperature=0)


    return confirmation["choices"][0]["text"]




def dictionary_present(response):
    delimiter = "####"
    user_req = {'Source': 'Delhi','Destination': 'Mumbai','Duration': '3','Budget': '20000'}
    prompt = f"""You are a python expert. You are provided an input.
            You have to check if there is a python dictionary present in the string.
            It will have the following format {user_req}.
            Your task is to just extract and return only the python dictionary from the input.
            The output should match the format as {user_req}.
            The output should contain the exact keys and values as present in the input.

            Here are some sample input output pairs for better understanding:
            {delimiter}
            input: - 'Source': 'Delhi','Destination': 'Mumbai','Duration': '3','Budget': '20000'
            output: {{'Source': 'Delhi','Destination': 'Mumbai','Duration': '3','Budget': '20000'}}

            input: - 'Source': 'Mumbai','Destination': 'Pune','Duration': '5','Budget': '50000'
            output: {{'Source': 'Mumbai','Destination': 'Pune','Duration': '5','Budget': '50000'}}

            input: - 'Source': 'Bangalore','Destination': 'Mumbai','Duration': '3','Budget': '10000'
            output: {{'Source': 'Bangalore','Destination': 'Mumbai','Duration': '3','Budget': '10000'}}
            
            {delimiter}

            Here is the input {response}

            """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens = 2000
        # temperature=0.3,
        # top_p=0.4
    )
    return response["choices"][0]["text"]



def extract_dictionary_from_string(string):
    regex_pattern = r"\{[^{}]+\}"

    dictionary_matches = re.findall(regex_pattern, string)

    # Extract the first dictionary match and convert it to lowercase
    if dictionary_matches:
        dictionary_string = dictionary_matches[0]
        dictionary_string = dictionary_string.lower()

        # Convert the dictionary string to a dictionary object using ast.literal_eval()
        dictionary = ast.literal_eval(dictionary_string)
    return dictionary




def fetch_travel_iternary(user_req_string):
    flight_data= pd.read_csv('flight_data.csv')
    print('user_req_string',user_req_string)
    user_requirements = extract_dictionary_from_string(user_req_string)
    print(user_requirements)

    budget = int(user_requirements.get('budget', '0').replace(',', '').split()[0])
    print('budget', budget)
    #This line retrieves the value associated with the key 'budget' from the user_requirements dictionary.

    filtered_flight_data_source = flight_data.copy()
    filtered_flight_data_source = filtered_flight_data_source[filtered_flight_data_source['source_city'].str.lower() == user_requirements['source']]
    filtered_flight_data_source = filtered_flight_data_source[filtered_flight_data_source['destination_city'].str.lower() == user_requirements['destination']]
    filtered_flight_data_source = filtered_flight_data_source[filtered_flight_data_source['price'] <= budget].copy()
    #These lines create a copy of the flight_data dataframe and assign it to filtered_flight_data.
    filtered_flight_data_destination = flight_data.copy()
    filtered_flight_data_destination = filtered_flight_data_destination[filtered_flight_data_destination['source_city'].str.lower() == user_requirements['destination']]
    filtered_flight_data_destination = filtered_flight_data_destination[filtered_flight_data_destination['destination_city'].str.lower() == user_requirements['source']]
    filtered_flight_data_destination = filtered_flight_data_destination[filtered_flight_data_destination['price'] <= budget].copy()
    
    print('filtered_flight_data_source', filtered_flight_data_source)
    print('filtered_flight_data_destination', filtered_flight_data_destination)
    
    df_merged = pd.concat([filtered_flight_data_source, filtered_flight_data_destination], ignore_index=True, sort=False)

    return df_merged.to_json(orient='records')




def recommendation_validation(travel_to_destination_flights):
    data = json.loads(travel_to_destination_flights)

    return data




def initialize_conv_reco(flights):
    system_message = f"""
    You are an intelligent travel agent and you are tasked with the objective to \
    recommend user travel plans based on their requirements: {flights}.\
    You should keep the user profile in mind while answering the questions.\

    Start with a brief summary of each flight data in the following format:
    1. <flight> : <source>, <destination>, <budget in Rs>, <departure_time>, <duration>
    2. <flight> : <source>, <destination>, <budget in Rs>, <departure_time>, <duration>

    """
    conversation = [{"role": "system", "content": system_message }]
    return conversation