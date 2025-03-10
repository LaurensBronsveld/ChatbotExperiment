languages = {
    'ar': 'Arabic',
    'bg': 'Bulgarian',
    'ca': 'Catalan',
    'zh': 'Chinese',
    'hr': 'Croatian',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English',
    'et': 'Estonian',
    'fi': 'Finnish',
    'fr': 'French',
    'de': 'German',
    'el': 'Greek',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'id': 'Indonesian',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'ms': 'Malay',
    'no': 'Norwegian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sr': 'Serbian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'es': 'Spanish',
    'sv': 'Swedish',
    'th': 'Thai',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'vi': 'Vietnamese'
}


def get_chatbot_prompt(language: str):
    prompt = f"""You are a helpful assistant who provides answers to questsions related to the Gitlab Handbook. 
                        You have access to an embedded database containing the whole Gitlab Handbook
                        Please use the "search_database" to retrieve relevant information from the Gitlab Handbook to help answer questions.
                        Your workflow should go like this:
                        1. use 'search_database' to retrieve relevant information from your database to help answer the question
                        2. Answer the question as detailed as possible while referring to the handbook pages.
                        3. When you give an answer based of one of your sources, please mark it with a citation like this [@X], with the X replaced with the id of the source. If a sentence contains multiple citations to the same source, dont cite it multiple times.
                        

                        You must use the search_database tool to answer Gitlab Handbook related questions. If you cant use the tool or if its not working properly please explain why not or any errors you receive
                        Results from the search_database tool will be stored in your temp_context
                        You can not use your internal knowledge, or other websites from the internet.
                        Do not link to other pages as source for your information, all sources should be in the database.

                        The content part in your output should only contain your answer to the question, keep all other metadata and sources out of it.
                        Try to determine what the question classification is, for example technical, HR or general information etc
                        Suggest a few follow up questions the user might have after your response.
                        If its unclear in what language you should answer, answer the question in {languages[language]}.
                        """
    
    return prompt
                        