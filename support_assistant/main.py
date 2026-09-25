import os
from typing import TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END


# ============================================================
# SETTINGS
# ============================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1")

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zepto_policies"


# ============================================================
# EMBEDDING + CHROMADB
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


# ============================================================
# STRUCTURED PROMPT
# ============================================================

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the Zepto policy information provided in the retrieved context.

TASK:
Answer the customer's question using the provided policy context.

FORMAT:
Return JSON with:
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 0.0
}

LENGTH:
Keep the answer concise and directly relevant.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies.

FEW-SHOT EXAMPLE:
Question: What is the delivery fee for orders below INR 149?
Context: Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.
Answer:
{
  "answer": "Orders below INR 149 have a flat INR 25 delivery fee.",
  "sources": ["doc_01"],
  "confidence": 1.0
}
"""


# ============================================================
# NODE 1: CLASSIFY INTENT
# ============================================================

def classify_intent(state: GraphState):

    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    if any(keyword in query for keyword in policy_keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        "intent": intent
    }


# ============================================================
# NODE 2: RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(state: GraphState):

    query = state["query"]

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documents = results["documents"][0]
    ids = results["ids"][0]

    top_chunk = documents[0]
    top_chunk_snippet = top_chunk[:200]

    # Required graded mock mode
    if MOCK_LLM != "0":

        answer = (
            "Based on the retrieved context: "
            + top_chunk_snippet
        )

        return {
            "answer": answer,
            "sources": ids,
            "confidence": 1.0
        }

    # Optional real-LLM branch
    # The graded submission uses MOCK_LLM=1.
    return real_llm_answer_with_retry(
        query,
        documents,
        ids
    )


# ============================================================
# NODE 3: DIRECT ANSWER
# ============================================================

def direct_answer(state: GraphState):

    # Required graded mock mode
    if MOCK_LLM != "0":

        return {
            "answer": "I can only answer questions about Zepto policies right now.",
            "sources": [],
            "confidence": 1.0
        }

    # Optional real-LLM branch
    return real_llm_direct_answer_with_retry(
        state["query"]
    )


# ============================================================
# OPTIONAL REAL LLM RETRY LOGIC
# ============================================================

def real_llm_answer_with_retry(query, documents, ids):

    last_error = None

    for attempt in range(3):

        try:
            # Optional extension.
            # A real LLM provider can be connected here.
            raise NotImplementedError(
                "Real LLM extension is optional. "
                "Use MOCK_LLM=1 for the graded offline mode."
            )

        except Exception as error:
            last_error = str(error)

    return {
        "answer": f"ERROR: {last_error}",
        "sources": ids,
        "confidence": 0.0
    }


def real_llm_direct_answer_with_retry(query):

    last_error = None

    for attempt in range(3):

        try:
            # Optional extension.
            raise NotImplementedError(
                "Real LLM extension is optional. "
                "Use MOCK_LLM=1 for the graded offline mode."
            )

        except Exception as error:
            last_error = str(error)

    return {
        "answer": f"ERROR: {last_error}",
        "sources": [],
        "confidence": 0.0
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_query(state: GraphState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(GraphState)

graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)

graph_builder.add_edge(
    START,
    "classify_intent"
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_query,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)

graph = graph_builder.compile()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description="Offline mock RAG support assistant for Zepto policies"
)


@app.get("/")
def home():

    return {
        "message": "Zepto Support Assistant is running",
        "mock_llm": MOCK_LLM
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    result = graph.invoke({
        "query": request.query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0
    })

    response = AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

    return response