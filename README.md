# Varco LLM을 이용하여 간단한 한국어 Chatbot 구현하기

Varco LLM의 입력과 출력의 포맷을 맞추어서 아래와 같이 ContentHandler를 정의합니다. 

```python
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
```

Varco LLM은 SageMaker endpoint를 이용하여 접근할 수 있습니다. 아래와 같이 ContentHandler를 이용하여 LangChain을 연결합니다.

```python
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
```

## 문서 읽기

S3에서 PDF, TXT, CSV 파일을 아래처럼 읽어올 수 있습니다. pdf의 경우에 PyPDF2를 이용하여 PDF파일에서 page 단위로 읽어옵니다. 이때, 불필요한 '\x00', '\x01'은 아래와 같이 제거합니다. 또한 LLM의 token size 제한을 고려하여, 아래와 같이 RecursiveCharacterTextSplitter을 이용하여 chunk 단위로 문서를 나눕니다. 

```python
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
    
    new_contents = str(contents).replace("\n"," ") 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    texts = text_splitter.split_text(new_contents) 
            
    return texts
```

## 답변하기

Varco는 User의 요청을 같이 전달하고 응답은 "### Assistant:" 포맷으로 전달되므로, LLM의 응답에서 답변만 골라서 메시지로 전달합니다.

```python
answer = llm(text)
print('answer: ', answer)

pos = answer.rfind('### Assistant:\n') + 15
msg = answer[pos:]
```

## 읽어온 문서를 Document로 저장하기

아래와 같이 load_document()를 이용하여 S3로 부터 읽어온 문서를 page 단위로 Document()에 저장합니다. 이때 파일이름과 Chunk의 순서를 가지고 metadata를 아래와 같이 정의합니다. 
```python
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
```

사용자의 편의를 위하여 아래와 같이 읽어온 문서의 3page 내에서 문서의 요약 정보를 제공합니다.

```python
prompt_template = """Write a concise summary of the following:

{ text }
                
CONCISE SUMMARY """

PROMPT = PromptTemplate(template = prompt_template, input_variables = ["text"])
chain = load_summarize_chain(llm, chain_type = "stuff", prompt = PROMPT)
summary = chain.run(docs)

pos = summary.rfind('### Assistant:\n') + 15
msg = summary[pos:]
```



## Reference

LangChain없이 API를 이용하여 아래와 같이 응답을 얻을 수 있습니다.

```python
payload = {
    "text": text,
    "request_output_len": 512,
    "repetition_penalty": 1.1,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.9
}

client = boto3.client('runtime.sagemaker')
response = client.invoke_endpoint(
    EndpointName = endpoint_name,
    ContentType = 'application/json',
    Body = json.dumps(payload).encode('utf-8'))

response_payload = json.loads(response['Body'].read())

msg = response_payload['result'][0]
```
