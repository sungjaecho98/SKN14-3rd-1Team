from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def nutrition_recommend(prompt, model='gpt-4o-mini', temperature=1, top_p=1):
    system_message = """
    
    
    
    """
    user_message = f"""
    사용자 현재상황:
    {prompt}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_message
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message
                    }
                ]
            }],
        response_format={
            "type": "text"
        },
        temperature=temperature,
        max_completion_tokens=2048,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content
