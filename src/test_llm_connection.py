import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def test_langchain_connection():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("❌ No API key found. Add it to your .env file.")

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Prompt to verify LLM access
    prompt = PromptTemplate.from_template(
        "Explain SHAP in one sentence for a data science hackathon."
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({})

    print("✅ LLM Response:\n", response)

if __name__ == "__main__":
    test_langchain_connection()
