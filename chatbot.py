import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

def create_qa_chain():
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    llm = ChatGroq(
      model="qwen/qwen3.6-27b",
      temperature=0.1,
      groq_api_key=os.getenv("GROQ_API_KEY")
    )

    prompt_template = """Tum ek expert tutor ho jo practice sheet ke questions solve karte ho.

Neeche diya gaya context tumhari practice sheet ka content hai. Isme se relevant question dhundo aur use khud apni knowledge aur reasoning use karke solve karo — sahi answer step-by-step explanation ke saath do.

Agar question multiple choice (options) wala hai, to correct option (A/B/C/D) batao aur reasoning bhi do.
Agar question bina options ka hai, to direct sahi answer nikal ke do, calculation/reasoning ke saath.

LANGUAGE RULES (bahut important, follow karo):
- Agar user ka sawal pure English me hai, to poora answer pure English me do.
- Agar user ka sawal pure Hindi (Devanagari script) me hai, to poora answer pure Hindi me do.
- Agar user ka sawal Hinglish (Hindi+English mix, Roman script) me hai, to answer bhi Hinglish me hi do.
- Chahe jis bhi language/style me jawab do, answer ki shuruaat hamesha ek English word ya short English phrase se karo (jaise "Answer:", "Sure,", "Okay," jaisa kuch), uske baad baaki poora jawab us user ki language me continue karo.

Context (practice sheet se related portion): {context}

Question: {question}

Answer (upar diye gaye LANGUAGE RULES ke hisaab se, step-by-step solve karke):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

if __name__ == "__main__":
    qa = create_qa_chain()
    print("Chatbot ready! (type 'exit' to quit)\n")

    while True:
        query = input("Tumhara sawal: ")
        if query.lower() == "exit":
            break
        result = qa.invoke({"query": query})
        print("\nJawab:", result["result"])
        print("\nSource:", [doc.metadata.get("source") for doc in result["source_documents"]])
        print("-" * 50)