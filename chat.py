import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loader =PyPDFLoader(file_path=r".\scet.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(data)

#initialize large Language Model for answer generation
llm_answer_gen = LlamaCpp(
    streaming=True,
    model_path=r"./mistral-7b-openorca.Q4_0.gguf",
    temperature=0.75,
    top_p=1,
    f16_kv=True,
    verbose=False,
    n_ctx=4096
    )

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

vector_store = Chroma.from_documents(text_chunks, embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
answer_gen_chain = ConversationalRetrievalChain.from_llm(llm=llm_answer_gen, retriever=vector_store.as_retriever(), memory=memory)

while True:
    user_input = input("Enter a question: ")
    if user_input.lower() == "q":
        break

    answers = answer_gen_chain.run({"question": user_input})
    print("Answer: ", answers)

