import json
import boto3
import os
import time
import datetime
from io import BytesIO
import PyPDF2
import csv
import sys

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from langchain.vectorstores import FAISS
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.document_loaders import CSVLoader
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import SagemakerEndpointEmbeddings

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
endpoint_name = os.environ.get('endpoint_name')

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
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.9
} 

llm = SagemakerEndpoint(
    endpoint_name = endpoint_name, 
    region_name = aws_region, 
    model_kwargs = parameters,
    endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    content_handler = content_handler
)

# load documents from s3
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
        contents.replace('\x00', '')

        print('contents: ', contents)
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
    elif file_type == 'csv':        
        body = doc.get()['Body'].read().decode('utf-8')
        reader = csv.reader(body)        
        contents = CSVLoader(reader)
    
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
            
    return texts
              
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

    start = int(time.time())    

    msg = ""
    
    if type == 'text':
        text = body

        """
        payload = {
            "text": text,
            "request_output_len": 512,
            #"repetition_penalty": 1.3,
            #"temperature": 0.5
            "repetition_penalty": 1.1,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 0.9
        }

        client = boto3.client('runtime.sagemaker')
        response = client.invoke_endpoint(
            EndpointName=endpoint_name, 
            ContentType='application/json', 
            Body=json.dumps(payload).encode('utf-8'))                        
        #print('response:', response)
        response_payload = json.loads(response['Body'].read())
        print('response_payload:', response_payload)

        msg = response_payload['result'][0]
        
        """
        
        answer = llm(text)
        print('answer: ', answer)

        pos = answer.rfind('### Assistant:\n')+15
        msg = answer[pos:]    
            
    elif type == 'document':
        object = body
        
        file_type = object[object.rfind('.')+1:len(object)]
        print('file_type: ', file_type)
            
        # load documents where text, pdf, csv are supported
        texts = load_document(file_type, object)

        docs = []
        for i in range(len(texts)):
            docs.append(
                Document(
                    page_content=texts[i],
                    metadata={
                        'name': object,
                        'page':i+1
                    }
                )
            )        
        print('docs[0]: ', docs[0])    
        print('docs size: ', len(docs))
        
        # summerization to show the document
        prompt_template = """Write a concise summary of the following:

        {text}
                
        CONCISE SUMMARY """

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
        summary = chain.run(docs)
        print('summary: ', summary)

        msg = summary
                
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
