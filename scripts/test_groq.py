
import os

from dotenv import load_dotenv
from groq import Groq


def main():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from .env")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Groq test successful",
            }
        ],
        temperature=0,
    )

    print("Model:", model)
    print("Response:", response.choices[0].message.content)


if __name__ == "__main__":
    main()
