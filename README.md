# Pygraph Using Prompts to Draw Models:

Create a python graphviz diagram that shows this graph:  1. File Uploader Converts Input Files to Text, 2. Extract 3 Text Chunks and Create an Embedding for Each.  3. Combine embeddings into AI Semantic Index (FAISS) and Store as Knowledge Base (KB) 4. Add Semantic Search with Question and Question Embeddings.  5. Put both query embeddings and Document embeddings onto Chat Memory on a Language Chain then run Inference.  5. Store inference output and show to user.  6. For output score Rouge LCS to define longest continuous match with original document.  7. Give user a way to upvote using thumbs up emoji or downvote thumbs down emoji the combination of embeddings for document context and question context as pair then name files for each step with LCS response.

```
import streamlit as st
from graphviz import Digraph

# Initialize Streamlit app
st.title("Process Flow Diagram")

# Initialize Graphviz object
dot = Digraph(format="png")
dot.attr(rankdir='LR', size='10')

# Add nodes and edges
dot.node("1", "File Uploader\nConverts Input Files to Text")
dot.node("2", "Extract 3 Text Chunks\nand Create an Embedding for Each")
dot.node("3", "Combine Embeddings into\nAI Semantic Index (FAISS)\nand Store as Knowledge Base (KB)")
dot.node("4", "Add Semantic Search with\nQuestion and Question Embeddings")
dot.node("5", "Put Query and Document\nEmbeddings onto Chat Memory\non a Language Chain then Run Inference")
dot.node("6", "Store Inference Output\nand Show to User")
dot.node("7", "Score Rouge LCS to Define\nLongest Continuous Match with\nOriginal Document")
dot.node("8", "Give User a Way to Upvote 👍\nor Downvote 👎\nand Name Files for Each Step")

# Define edges
dot.edges(["12", "23", "34", "45", "56", "67", "78"])

# Generate and save the diagram
dot.render(filename="/tmp/process_flow", cleanup=True)

# Display using Streamlit
st.image("/tmp/process_flow.png")
```

```
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import faiss
import random

st.title("NLP Workflow Simulation")

# Step 1: File Uploader Converts Input Files to Text
uploaded_file = st.file_uploader("Step 1: Upload a text file", type=["txt"])

if uploaded_file is not None:
    uploaded_text = uploaded_file.read().decode('utf-8')
    st.write("Uploaded text:", uploaded_text[:100] + "...")

    # Step 2: Extract 3 Text Chunks and Create an Embedding for Each
    chunks = uploaded_text.split()[:9]  # Simplified chunk extraction
    chunks = [' '.join(chunks[i:i + 3]) for i in range(0, len(chunks), 3)]
    st.write(f"Step 2: Extracted Text Chunks: {chunks}")

    # Placeholder for creating embeddings
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(chunks)

    # Step 3: Combine Embeddings into AI Semantic Index (FAISS) and Store as KB
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings.todense()).astype('float32'))
    st.write("Step 3: Created FAISS index and stored as Knowledge Base")

    # Step 4: Add Semantic Search with Question and Question Embeddings
    question = st.text_input("Step 4: Input a question for semantic search")

    if question:
        question_embedding = vectorizer.transform([question]).todense()
        _, idx = index.search(np.array(question_embedding).astype('float32'), 1)
        st.write(f"Found closest chunk to question: {chunks[idx[0][0]]}")

        # Step 5: Put Query and Document Embeddings onto Chat Memory and Run Inference
        # Placeholder for inference
        inference_output = "Inferred answer: " + random.choice(chunks)
        st.write(f"Step 5: {inference_output}")

        # Step 6: Store Inference Output and Show to User
        st.write("Step 6: Stored inference output and showing to user")

        # Step 7: Score Rouge LCS
        # Placeholder for Rouge LCS scoring
        rouge_score = random.uniform(0, 1)
        st.write(f"Step 7: Rouge LCS score is {rouge_score:.2f}")

        # Step 8: Give User a Way to Upvote or Downvote
        feedback = st.radio("Step 8: Give your feedback", ("👍 Upvote", "👎 Downvote"))

        if feedback == "👍 Upvote":
            st.write("You upvoted 👍. Thank you for your feedback!")
        else:
            st.write("You downvoted 👎. Thank you for your feedback!")
```

# Mermaid Model
Create the graph model using mermaid instead.  show image of the model

graph LR
  A[File Uploader Converts Input Files to Text] --> B[Extract 3 Text Chunks and Create an Embedding for Each]
  B --> C[Combine Embeddings into AI Semantic Index (FAISS) and Store as Knowledge Base (KB)]
  C --> D[Add Semantic Search with Question and Question Embeddings]
  D --> E[Put Query and Document Embeddings onto Chat Memory on a Language Chain then Run Inference]
  E --> F[Store Inference Output and Show to User]
  F --> G[Score Rouge LCS to Define Longest Continuous Match with Original Document]
  G --> H[Give User a Way to Upvote or Downvote and Name Files for Each Step]








# ChatGPTCustomInstructions

I live in ___ and work for a ___ company and I write AI programs and invent new programs that are innovative and help people ___.  
My hobbies are swimming, programming, gaming, gymnast rings, teaching AI, and taking care of my ___, my ___ __ and __ our new ___.  
I can talk about AI for hours, possibly also health care, medical, behavioral, psychology, and learning and perception theory.  
Some goals I have include ___, being a great teacher of __, having a happy family life, and staying healthy by exercising my body and brain and learning as much as I can each day by pair programming with AI!

Respond with mixture of expert roles below.  Respond according to my CHARMSEW acronym when more than one MoE role is used.  When generating code always show full listing.  Never omit code lines where you replace content with comments.  Favor no comments and all code instead.

1. C is for Coder:  Write streamlit python apps, Node.js, Javascript, HTML5, and C# with Typescript.  Long full code listing
2. H is for Humanities: Write as expert in humanities with long responses with details.  Include emojis in every line for expression.
3. A is for Analysis: Favor doing logical analysis and spelling out the method steps.  model or graph generation python code for analysis.
4. R is for Roleplay: Use stepwise features and numbered outlines and list all people and roles mentioned in context.
5. M is for Math: instruction steps numbered with insight about concepts related areas.
6. S is for STEM:  Be an expert on Science, Technology, Engineering and Math.  Operate post doctoral level.  Provide definitions and list keywords, useful links.
7. E is for Extraction:  When doing extraction try to follow some standards and also let me know what is being extracted and how those terms match up with my prompt. 
8. W is for Writing:  Write using new words in appropriate context.  When writing technical be logical.  When writing text use adjectives, metaphors and parts of speech.  Add appropriate emojis to writing  around any word you can for an easy view of appropriate emoji for the word.
