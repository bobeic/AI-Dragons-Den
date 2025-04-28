import os
import faiss
import pickle
from uuid import uuid4
from typing import Callable, Any, Optional
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_dragon_vector_store(
    dragon_name: str,
    texts: list[str],
    metadata_fn: Optional[Callable[[str], dict]] = None,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    index_factory: Callable[[int], Any] = lambda dim: faiss.IndexFlatL2(dim),
    save_path: str = "./vectorstores",
):
    if metadata_fn is None:
        metadata_fn = lambda txt: {"source": dragon_name}

    # Paths
    dragon_folder = os.path.join(save_path, dragon_name)
    index_file = os.path.join(dragon_folder, "index.faiss")
    docstore_file = os.path.join(dragon_folder, "docstore.pkl")

    # Try loading
    if os.path.exists(index_file) and os.path.exists(docstore_file):
        print(f"Loading vectorstore for {dragon_name} from disk...")
        index = faiss.read_index(index_file)
        with open(docstore_file, "rb") as f:
            docstore = pickle.load(f)

        embeds = HuggingFaceEmbeddings(model_name=embedding_model)
        vs = FAISS(
            embedding_function=embeds,
            index=index,
            docstore=docstore,
            index_to_docstore_id={i: str(i) for i in range(index.ntotal)},
        )
        return vs

    # Else create fresh
    print(f"Creating new vectorstore for {dragon_name}...")
    embeds = HuggingFaceEmbeddings(model_name=embedding_model)
    dim = len(embeds.embed_query(texts[0]))
    index = index_factory(dim)

    vs = FAISS(
        embedding_function=embeds,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    docs = [Document(page_content=txt, metadata=metadata_fn(txt)) for txt in texts]
    ids = [str(uuid4()) for _ in docs]
    vs.add_documents(documents=docs, ids=ids)

    # Save to disk
    os.makedirs(dragon_folder, exist_ok=True)
    faiss.write_index(vs.index, index_file)
    with open(docstore_file, "wb") as f:
        pickle.dump(vs.docstore, f)

    print(f"Saved vectorstore for {dragon_name} to {dragon_folder}")
    return vs
