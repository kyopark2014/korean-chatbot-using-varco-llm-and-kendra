import json
import boto3
import os
import time
import datetime
from io import BytesIO
import PyPDF2
import csv

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
endpoint_name = os.environ.get('endpoint_name')
varco_region = os.environ.get('varco_region')
kendraIndex = os.environ.get('kendraIndex')
roleArn = os.environ.get('roleArn')
enableKendra = os.environ.get('enableKendra')
enableReference = os.environ.get('enableReference')
enableRAG = os.environ.get('enableRAG', 'true')

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({
            "text" : prompt, **model_kwargs
        })
        return input_str.encode('utf-8')
      
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["result"][0]

content_handler = ContentHandler()
aws_region = boto3.Session().region_name
client = boto3.client("sagemaker-runtime")
parameters = {
    "request_output_len": 512,
    "repetition_penalty": 1.1,
    "temperature": 0.1,
    "top_k": 50,
    "top_p": 0.1
} 

llm = SagemakerEndpoint(
    endpoint_name = endpoint_name, 
    region_name = varco_region, 
    model_kwargs = parameters,
    endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    content_handler = content_handler
)

# memory for retrival docs
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question", output_key='answer', human_prefix='Human', ai_prefix='AI')
# memory for conversation
chat_memory = ConversationBufferMemory(human_prefix='Human', ai_prefix='AI')

retriever = AmazonKendraRetriever(index_id=kendraIndex)

# store document into Kendra
def store_document(s3_file_name, requestId):
    documentInfo = {
        "S3Path": {
            "Bucket": s3_bucket,
            "Key": s3_prefix+'/'+s3_file_name
        },
        "Title": s3_file_name,
        "Id": requestId
    }

    documents = [
        documentInfo
    ]

    kendra = boto3.client("kendra")
    result = kendra.batch_put_document(
        Documents = documents,
        IndexId = kendraIndex,
        RoleArn = roleArn
    )
    print(result)

# load documents from s3
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            page_text = page.extract_text().replace('\x00','')
            raw_text.append(page_text.replace('\x01',''))
        contents = '\n'.join(raw_text)            
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
    elif file_type == 'csv':        
        body = doc.get()['Body'].read().decode('utf-8')
        reader = csv.reader(body)        
        contents = CSVLoader(reader)
    
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
            
    return texts

def summerize_text(text):
    docs = [
        Document(
            page_content=text
        )
    ]
    prompt_template = """다음 텍스트를 간결하게 요약하십시오. 
    
    TEXT: {text}
                
    SUMMARY:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    summary = chain.run(docs)
    print('summarized text: ', summary)

    return summary

def get_reference(docs):
    reference = "\n\nFrom\n"
    for doc in docs:
        name = doc.metadata['title']
        if(doc.metadata['document_attributes'] != {}):
            page = doc.metadata['document_attributes']['_excerpt_page_number']
            reference = reference + f"{page}page in {name}\n"
        else:
            reference = reference + f"in {name}\n"
    return reference

def get_answer_using_template_with_history(query, chat_memory):  
    condense_template = """아래의 대화 내용을 고려하여 친구처럼 친절하게 대답해줘. 새로운 질문에만 대답하고, 모르면 모른다고 해.
    
    {chat_history}
    
    User: {question}
    Assistant:"""    
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever,       
        condense_question_prompt=CONDENSE_QUESTION_PROMPT, # chat history and new question
        chain_type='stuff', # 'refine'
        verbose=False, # for logging to stdout
        rephrase_question=True,  # to pass the new generated question to the combine_docs_chain
        
        memory=memory,
        #max_tokens_limit=300,
        return_source_documents=True, # retrieved source
        return_generated_question=False, # generated question
    )

    # combine any retrieved documents.
    prompt_template = """다음은 User와 Assistant의 친근한 대화입니다. 
Assistant은 말이 많고 상황에 맞는 구체적인 세부 정보를 많이 제공합니다. 
Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다.

    {context}

    Question: {question}
    Assistant:"""
    qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(prompt_template) 
    
    # extract chat history
    chats = chat_memory.load_memory_variables({})
    chat_history_all = chats['history']
    print('chat_history_all: ', chat_history_all)

    # use last two chunks of chat history
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=0)
    texts = text_splitter.split_text(chat_history_all) 

    pages = len(texts)
    print('pages: ', pages)

    if pages >= 2:
        chat_history = f"{texts[pages-2]} {texts[pages-1]}"
    elif pages == 1:
        chat_history = texts[0]
    else:  # 0 page
        chat_history = ""
    print('chat_history:\n ', chat_history)

    # make a question using chat history
    result = qa({"question": query, "chat_history": chat_history})    
    print('result: ', result)    
    
    # get the reference
    source_documents = result['source_documents']
    print('source_documents: ', source_documents)

    if len(source_documents)>=1 and enableReference == 'true':
        reference = get_reference(source_documents)
        #print('reference: ', reference)
        return result['answer']+reference
    else:
        return result['answer']

def get_answer_using_template(query):
    #relevant_documents = retriever.get_relevant_documents(query)
    #print('length of relevant_documents: ', len(relevant_documents))

    #if(len(relevant_documents)==0):
    #    return llm(query)
    #else:
    #    print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    #    print('----')
    #    for i, rel_doc in enumerate(relevant_documents):
    #        print(f'## Document {i+1}: {rel_doc.page_content}.......')
    #        print('---')

    prompt_template = """다음은 User와 Assistant의 친근한 대화입니다. 
Assistant은 말이 많고 상황에 맞는 구체적인 세부 정보를 많이 제공합니다. 
Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다.

    {context}

    Question: {question}
    Assistant:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = qa({"query": query})
    print('result: ', result)

    source_documents = result['source_documents']        
    print('source_documents: ', source_documents)
    
    if len(source_documents)>=1 and enableReference == 'true':
        reference = get_reference(source_documents)
        # print('reference: ', reference)
        return result['result']+reference
    else:
        return result['result']

def lambda_handler(event, context):
    print(event)
    userId  = event['user-id']
    print('userId: ', userId)
    requestId  = event['request-id']
    print('requestId: ', requestId)
    type  = event['type']
    print('type: ', type)
    body = event['body']
    print('body: ', body)

    global llm, kendra
    global enableRAG, enableReference, enableConversationMode

    start = int(time.time())    

    msg = ""
    
    if type == 'text':
        text = body

        querySize = len(text)
        print('query size: ', querySize)

        # debugging
        if text == 'enableReference':
            enableReference = 'true'
            msg  = "Referece is enabled"
        elif text == 'disableReference':
            enableReference = 'false'
            msg  = "Reference is disabled"
        elif text == 'enableConversationMode':
            enableConversationMode = 'true'
            msg  = "onversationMode is enabled"
        elif text == 'disableConversationMode':
            enableConversationMode = 'false'
            msg  = "onversationMode is disabled"
        elif text == 'enableRAG':
            enableRAG = 'true'
            msg  = "RAG is enabled"
        elif text == 'disableRAG':
            enableRAG = 'false'
            msg  = "RAG is disabled"
        else:
            
            if querySize<1000 and enableRAG=='true': 
                if enableConversationMode == 'true':
                    answer = get_answer_using_template_with_history(text, chat_memory)
                else:
                    answer = get_answer_using_template(text)
            else:
                answer = llm(text)        
            print('answer: ', answer)

            pos = answer.rfind('### Assistant:\n')+15
            msg = answer[pos:]    
        #print('msg: ', msg)
        chat_memory.save_context({"input": text}, {"output": msg})
            
    elif type == 'document':
        object = body

        # stor the object into kendra
        store_document(object, requestId)
        
        file_type = object[object.rfind('.')+1:len(object)]
        print('file_type: ', file_type)
            
        # summerization to show the document
        texts = load_document(file_type, object)
        docs = [
            Document(
                page_content=t
            ) for t in texts[:3]
        ]
        
        prompt_template = """다음 텍스트를 간결하게 요약하십시오. 
    
        TEXT: {text}
                
        SUMMARY:"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
        summary = chain.run(docs)
        print('summary: ', summary)

        pos = summary.rfind('### Assistant:\n')+15
        msg = summary[pos:] 
                
    elapsed_time = int(time.time()) - start
    print("total run time(sec): ", elapsed_time)

    print('msg: ', msg)

    item = {
        'user-id': {'S':userId},
        'request-id': {'S':requestId},
        'type': {'S':type},
        'body': {'S':body},
        'msg': {'S':msg}
    }

    client = boto3.client('dynamodb')
    try:
        resp =  client.put_item(TableName=callLogTableName, Item=item)
    except: 
        raise Exception ("Not able to write into dynamodb")
        
    print('resp, ', resp)

    return {
        'statusCode': 200,
        'msg': msg,
    }
