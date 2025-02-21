# This simple crew  shows an agent that can summarize an article and write a linkedin post
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool
from crewai import LLM

os.environ["AZURE_API_KEY"] = "[INSERT CORRECT API KEY SEE SLIDES]"
os.environ["AZURE_API_BASE"] = "https://oai-nncrewai.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "2024-08-01-preview"


llm = LLM(model="azure/gpt-4o", temperature=0.7)

# this could have been a parameter
website_url = "https://www.novonesis.com/en/news/biosolutions-waste-based-biodiesel"
web_reader_tool = ScrapeWebsiteTool(website_url=website_url)

web_researcher_agent = Agent(
    role="Web Researcher",
    goal="Conduct analysis of research articles and summarize for other people",
    backstory="You are a thorough researcher that has deep knowledge of biosolutions and are good at deducting and summarize information in a way that others can understand it",
    verbose=True,
    llm=llm,
    tools=[web_reader_tool],
    allow_delegation=False,
)

summarize_article_task = Task(
    description="Analyze a complex technical architectle. Summarize information that is relevant for a short linkedin post about the topcis that has the purpose of getting people to read the article",
    agent=web_researcher_agent,
    expected_output="A summary of an article that can be used to write linkedin post",
)


linkedin_writer_agent = Agent(
    role="Linked Writer",
    goal="Write awesome linkedin post that many people will read and like",
    backstory="You are a specialist in social media and are good at writing linkedin posts that many people click on. You have a gift for taking technical content and summarize it in an engaging way",
    verbose=True,
    llm=llm,
    allow_delegation=False,
)

write_linkedin = Task(
    description="Using the article summary provided to write a linkedin post in no more than {postcharacters} characters. The post must have a captivating title and a call to action to read the article",
    agent=linkedin_writer_agent,
    context=[summarize_article_task],
    expected_output="A completed linkedin post  in no more than {postcharacters} characters",
)

crew = Crew(
    agents=[
        web_researcher_agent,
    ],
    tasks=[
        summarize_article_task,
    ],
    verbose=True,
    process=Process.sequential,
)

inputs = {
    "postcharacters": "3000",
}
result = crew.kickoff(inputs=inputs)

print("***************** HERE IS YOUR LINKEDIN POST *****************")
print(result)
