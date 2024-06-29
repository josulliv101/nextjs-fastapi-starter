from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.schema.runnable import Runnable
from langchain_openai import ChatOpenAI
import os

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:

# How many dishes are burgers?
MATCH (d:Dish)-[:TYPE_OF]->(fc:FoodCategory)
WHERE apoc.text.levenshteinDistance(fc.name, "burger") < 2
RETURN COUNT(d) AS numberOfBurgerDishes

# What types of food does Burtons Grill offer?
MATCH (p:Place)
WHERE apoc.text.levenshteinDistance(p.name, "Burtons Grill") < 2
MATCH (p)-[:SERVES]->(d:Dish)-[:TYPE_OF]->(fc:FoodCategory)
RETURN DISTINCT fc.name AS foodTypesOffered

# What places have outdoor seating?
CALL db.index.fulltext.queryNodes("placeDescriptionIndex", "outdoor seating") YIELD node, score
RETURN node.name AS placeName, node.description AS placeDescription, score
ORDER BY score DESC

# What dishes have bacon?
CALL db.index.fulltext.queryNodes("dishDescriptionIndex", "bacon") YIELD node, score
RETURN node.name AS dishName, node.description AS dishDescription, score
ORDER BY score DESC

# List all steaks ordered by price.
MATCH (p:Place)-[:SERVES]->(d:Dish)-[:TYPE_OF]->(c:FoodCategory)
WHERE apoc.text.levenshteinDistance(c.name, "steak") < 2
RETURN p.name AS placeName, d.name AS steakName, d.price AS steakPrice
ORDER BY d.price DESC

# List all burgers ordered by price.
MATCH (p:Place)-[:SERVES]->(d:Dish)-[:TYPE_OF]->(c:FoodCategory)
WHERE apoc.text.levenshteinDistance(c.name, "burger") < 2
RETURN p.name AS placeName, d.name AS burgerName, d.price AS burgerPrice
ORDER BY d.price DESC

# List all the places that have burgers?
MATCH (p:Place)-[:SERVES]->(d:Dish)-[:TYPE_OF]->(fc:FoodCategory)
WHERE apoc.text.levenshteinDistance(fc.name, "burger") < 2
RETURN DISTINCT p.name AS placesWithBurgers

# List all the types of food available.
MATCH (fc:FoodCategory)
RETURN DISTINCT fc.name AS foodTypes

# Which places have both pizza and burgers?
MATCH (p:Place)-[:SERVES]->(d1:Dish)-[:TYPE_OF]->(fc1:FoodCategory)
WHERE apoc.text.levenshteinDistance(fc1.name, "burger") < 2
MATCH (p)-[:SERVES]->(d2:Dish)-[:TYPE_OF]->(fc2:FoodCategory)
WHERE apoc.text.levenshteinDistance(fc2.name, "pizza") < 2
RETURN DISTINCT p.name AS placesWithPizzaAndBurgers


The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

examples = [
    {
        "question": "How many dishes are burgers?",
        "query": """
        MATCH (d:Dish)-[:TYPE_OF]->(fc:FoodCategory)
        WHERE apoc.text.levenshteinDistance(fc.name, "burger") < 2 // Adjust the threshold as needed
        RETURN COUNT(d) AS numberOfBurgerDishes
        """,
    },
    {
        "question": "What types of food does Burtons Grill offer?",
        "query": """
        MATCH (p:Place)
        WHERE apoc.text.levenshteinDistance(p.name, "Burtons Grill") < 2 // Adjust the threshold as needed
        MATCH (p)-[:SERVES]->(d:Dish)-[:TYPE_OF]->(fc:FoodCategory)
        RETURN DISTINCT fc.name AS foodTypesOffered
        """,
    },
    {
        "question": "What places have outdoor seating?",
        "query": """
        CALL db.index.fulltext.queryNodes("placeDescriptionIndex", "outdoor seating") YIELD node, score
        RETURN node.name AS placeName, node.description AS placeDescription, score
        ORDER BY score DESC
        """,
    },
    {
        "question": "What dishes have bacon?",
        "query": """
        CALL db.index.fulltext.queryNodes("dishDescriptionIndex", "bacon") YIELD node, score
        RETURN node.name AS dishName, node.description AS dishDescription, score
        ORDER BY score DESC
        """,
    },
    {
        "question": "List all steaks ordered by price.",
        "query": """
        MATCH (p:Place)-[:SERVES]->(d:Dish)-[:TYPE_OF]->(c:FoodCategory)
        WHERE apoc.text.levenshteinDistance(c.name, "steak") < 2 // Adjust the threshold as needed
        RETURN p.name AS placeName, d.name AS steakName, d.price AS steakPrice
        ORDER BY d.price DESC
        """,
    },
    {
        "question": "List all burgers ordered by price.",
        "query": """
        MATCH (p:Place)-[:SERVES]->(d:Dish)-[:TYPE_OF]->(c:FoodCategory)
        WHERE apoc.text.levenshteinDistance(c.name, "burger") < 2 // Adjust the threshold as needed
        RETURN p.name AS placeName, d.name AS burgerName, d.price AS burgerPrice
        ORDER BY d.price DESC
        """,
    },
    {
        "question": "List all the places that have burgers?",
        "query": """
        MATCH (p:Place)-[:SERVES]->(d:Dish)-[:TYPE_OF]->(fc:FoodCategory)
        WHERE apoc.text.levenshteinDistance(fc.name, "burger") < 2 // Adjust the threshold as needed
        RETURN DISTINCT p.name AS placesWithBurgers
        """,
    },
    {
        "question": "List all the types of food available.",
        "query": """
        MATCH (fc:FoodCategory)
        RETURN DISTINCT fc.name AS foodTypes
        """,
    },
    {
        "question": "Which places have both pizza and burgers?",
        "query": """
        MATCH (p:Place)-[:SERVES]->(d1:Dish)-[:TYPE_OF]->(fc1:FoodCategory)
        WHERE apoc.text.levenshteinDistance(fc1.name, "burger") < 2 // Adjust the threshold as needed
        MATCH (p)-[:SERVES]->(d2:Dish)-[:TYPE_OF]->(fc2:FoodCategory)
        WHERE apoc.text.levenshteinDistance(fc2.name, "pizza") < 2 // Adjust the threshold as needed
        RETURN DISTINCT p.name AS placesWithPizzaAndBurgers
        """,
    },
]


print("CYPHER_GENERATION_PROMPT", CYPHER_GENERATION_PROMPT)

def graph_chain() -> Runnable:

    NEO4J_URI = os.getenv("NEO4J_URI")
    # NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    LLM = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        # database=NEO4J_DATABASE,
        sanitize=True,
    )

    graph.refresh_schema()

    example_prompt = PromptTemplate.from_template(
        "User input: {question}\nCypher query: {query}"
    )

    prompt = FewShotPromptTemplate(
        examples=examples[:5],
        example_prompt=example_prompt,
        prefix="You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.\n\nHere is the schema information\n{schema}.\n\nAlways use lowercase when comparing text within a cypher.\n\nBelow are a number of examples of questions and their corresponding Cypher queries.",
        suffix="User input: {question}\nCypher query: ",
        input_variables=["question", "schema"],
    )

    # Official API doc for GraphCypherQAChain at: https://api.python.langchain.com/en/latest/chains/langchain.chains.graph_qa.base.GraphQAChain.html#
    graph_chain = GraphCypherQAChain.from_llm(
        cypher_llm=LLM,
        qa_llm=LLM,
        validate_cypher=True,
        cypher_prompt=prompt,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        # return_direct = True,
    )

    return graph_chain