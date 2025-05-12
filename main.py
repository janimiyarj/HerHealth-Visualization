import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig
from pydantic import SecretStr
import asyncio
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)

# Load .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Setup LLM
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-exp',
    api_key=SecretStr(api_key)
)

# Use real Chrome browser
browser = Browser(config=BrowserConfig(
    chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless=False
))

# Setup agent with clearer task
agent = Agent(
task = """
1. Visit https://www.quora.com/What-are-some-major-problems-faced-by-girls-during-their-periods
2. Act like an empathetic PCOS/PCOD doctor or researcher conducting interviews.
3. Scroll through 50 patient posts which are less than 5 years from now, the age group of the people should 22 to 28 years old at the time of posting, focusing on unique, personal or diagnoised experiences.
4. For each post, extract:
   - Username (Reddit handle)
   - Non-obvious period experience (describe anything specific or personal about their period)
   - Any demographic info (age, location, ethnicity, etc.)
   - Health problems or symptoms they confirmed or relate to
   - Health problems or treatments they question or doubt
   - Anything they learned or found surprising from others or their own experience
   - What they are currently doing to manage PCOS/PCOD
   - What they wish for if they had a "magic wand" to solve their PCOS challenges

5. Format each post in the following structure and give me a all the 50 list of these entries:

---
User: <Quora Username>  
Non-Obvious Period Experience: <Describe the unique or personal period experience>  

Key Demographics: <Female, Age, Country, Ethnicity if mentioned>  

Supported Health Problems: <Symptoms or problems they confirm or support>  

Disputed Health Problems: <Symptoms or treatments they question or challenge>  

Learned Something New: <What they learned or how their perspective changed>  

Current Management: <What they are doing now – diet, medication, lifestyle>  

Magic Wand: <If they had a magic wand, what they’d wish for in solving PCOS>  
---

Return up to 50 such entries. If a field is missing in the post, write "Not specified."
"""
,
    llm=llm,
    browser=browser
)

# Run it
async def main():
    print("🚀 Starting agent task...")
    result = await agent.run()

    print("✅ Agent finished the run.\n")
    print("📝 Formatted Interview Outputs:\n")

    try:
        # Get the last non-empty message from the agent's history
        final_response = result[-1].content.strip()
        print(final_response)

        # Optional: Save to file
        with open("reddit_pcos_interviews.txt", "w", encoding="utf-8") as f:
            f.write(final_response)
            print("\n📁 Output saved to: reddit_pcos_interviews.txt")

    except Exception as e:
        print(f"⚠️ Could not extract final message. Error: {e}")

    input("✅ Task done! Press Enter to close the browser...")
    await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
