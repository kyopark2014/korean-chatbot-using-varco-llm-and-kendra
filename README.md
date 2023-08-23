# VARCO LLM와 Amazon Kendra를 이용하여 한국어 Chatbot 만들기

여기서는 [VARCO LLM](https://ncsoft.github.io/ncresearch/varco-llm/)와 [Amazon Kendra](https://docs.aws.amazon.com/ko_kr/kendra/latest/dg/what-is-kendra.html)를 이용하여 Question/Answering을 위한 한국어 Chatbot을 구현하고자 합니다. VARCO LLM은 엔씨소프트(NC SOFT)에서 제공하는 대용량 언어 모델(LLM)입니다. VARCO LLM KO-13B-IST는 VARCO LLM KO-13B-FM의 파인튜닝 모델로서 Question and Answering, Summarization등 다양한 태스크에 활용할 수 있습니다. VARCO LLM은 [Amazon SageMaker](https://aws.amazon.com/marketplace/seller-profile?id=seller-tkuvdeznmi2w4)를 이용하여 쉽게 배포하여 사용할 수 있습니다.  대규모 언어 모델(LLM)이 학습할때에 포함되지 못은 특정 영역의 데이터(domain specific data)는 [RAG (Retrieval Augmented Generation) ](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html)와 같이 외부 문서 저장소의 Question에 관련된 문서를 읽어와서 Prompt에 활용하는 방식으로 강화될수 있습니다. Amazon Kendra는 자연어 처리 및 고급 기계 학습 알고리즘을 사용하여 데이터에서 검색 질문에 대한 답변을 얻는 지능형 검색 서비스로서 대규모 언어 모델에 RAG를 구현할 때 유용하게 활용될 수 있습니다. 여기서는 대규모 언어 모델을 위한 어플리케이션 개발 프레임워크인 [LangChain](https://www.langchain.com/)을 활용하여 어플리케이션을 개발하며, Amazon의 대표적인 [서버리스 서비스](https://aws.amazon.com/ko/serverless/)인 [Amazon Lambda](https://aws.amazon.com/ko/lambda/)로 서빙하는 인프라를 구축합니다. Amazon Lambda를 비롯한 인프라를 배포하고 관리하기 위하여 [Amazon CDK]를 활용합니다.

전체적인 Architecture는 아래와 같습니다. 사용자의 Question은 Query로 [Amazon CloudFront](https://aws.amazon.com/ko/cloudfront/)와 [Amazon API Gateway](https://aws.amazon.com/ko/api-gateway/)를 거쳐서, Lambda에 전달됩니다. Lambda는 Kendra로 Query를 전달하여 관련된 문서들의 발췌를 받은후에 VARCO LLM에 전달하여 답변을 얻습니다. 이후 답변은 사용자에게 전달되어 채팅화면에 표시됩니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-varco-llm-and-kendra/assets/52392004/b0d38264-3a65-4c89-8477-cb8f291b6ebf)


## LangChain과 연동하기 

LangChain은 LLM application의 개발을 도와주는 Framework으로 Question anc Answering, Summarization등 다양한 task에 맞게 Chain등을 활용하여 편리하게 개발할 수 있습니다. VARCO LLM은 SageMaker Endpoint로 배포되므로 아래와 같이 VARCO LLM의 입력과 출력의 포맷을 맞추어서 ContentHandler를 정의합니다. 

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

VARCO LLM은 SageMaker endpoint를 이용하여 접근할 수 있습니다. 아래와 같이 ContentHandler를 이용하여 LangChain을 연결합니다. 

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

VARCO LLM의 parameter는 아래와 같습니다.
- request_output_len: 생성되는 최대 token의 수, 기본값은 1000입니다.
- repetition_penalty: 반복을 제한하기 위한 파라미터로 1.0이면 no panalty입니다. 기본값은 1.3입니다.
- temperature: 다음 token의 확율(probability)로서 기본값은 0.5입니다.

Output은 Json 형태로 전달되며 기본 포맷은 아래와 같습니다.

```java
{
  "result": [
    "output text here"
  ]
}
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

VARCO는 User의 요청을 같이 전달하고 응답은 "### Assistant:" 포맷으로 전달되므로, LLM의 응답에서 답변만 골라서 메시지로 전달합니다.

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

### AWS CDK로 인프라 구현하기

[CDK 구현 코드](./cdk-qa-with-rag/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다.

## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)


### CDK를 이용한 인프라 설치
[인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 


### 실행결과

한글파일을 올리고 관련 질문을 하면 아래와 같이 파일을 참조하여 응답을 얻을 수 있습니다. 

![image](https://github.com/kyopark2014/korean-chatbot-using-varco-llm-and-kendra/assets/52392004/8a152ebd-8010-4918-b81c-4d579ec46018)

다른 질문으로 테스트할 경우에도 아래와 같이 답변합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-varco-llm-and-kendra/assets/52392004/95c2a9c3-56a4-4793-aad4-61a699e0b543)

### 리소스 정리하기

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. [Cloud9 console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)에 접속하여 아래와 같이 삭제를 합니다.

```java
cdk destroy
```

본 실습에서는 VARCO LLM의 endpoint로 "ml.g5.12xlarge"를 사용하고 있으므로, 더이상 사용하지 않을 경우에 반드시 삭제하여야 합니다. 특히 cdk destroy 명령어로 Chatbot만 삭제할 경우에 SageMaker Endpoint가 유지되어 지속적으로 비용이 발생될 수 있습니다. 이를 위해 Endpoint Console에 접속해서 Endpoint를 삭제합니다. 마찬가지로 Models과 Endpoint configuration에서 설치한 VARCO LLM의 Model과 Configuration을 삭제합니다.

## 결론

엔씨소프트의 한국어 언어모델인 VARCO LLM과 Amazon Kendra를 활용하여 질문과 답변(Question/Answering) 테스크를 수행하는 Chatbot 어플리케이션을 구현하였습니다. 대규모 언어 모델(LLM)을 활용하면 기존 Rule 기반의 Chatbot보다 훨씬 강화된 기능을 제공할 수 있습니다. 대규모 언어모델 확습에 포함되지 못한 특정 영역의 데이터는 Amazon Kendra를 통해 보완될수 있으며, 이를 통해 엔터프라이즈 기업과 같이 질문과 답변을 고객에게 제공하는 기업들에 유용하게 사용될 수 있을것으로 보여집니다. 또한 대규모 언어 모델을 개발하는 프레임워크인 LangChain을 VARCO LLM과 연동하는 방법과 Amazon Kendra와 관련된 서빙 인프라를 AWS CDK를 활용하여 쉽게 구현할 수 있었습니다. 한국어 대규모 언어 모델은 Chatbot뿐 아니라 향후 다양한 분야에서 유용하게 활용될 수 있을것으로 기대됩니다.

## Reference

[NC - github](https://ncsoft.github.io/ncresearch/varco-llm/)

[Deploy VARCO LLM Model 13B IST Package from AWS Marketplace](https://github.com/ncsoft/ncresearch/blob/main/notebooks/varco_model_13_IST.ipynb)

### invode_endpoint API 사용 예제
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


### Troubleshooting

#### Error: AttributeError: 'kendra' object has no attribute 'retrieve'

SageMaker Endpoint를 이용할때에 LangChain으로 kendra의 retriever를 정의할때 아래와 같은 에러가 발생하였습니다. 결과적으로 Dockerfile의 Python version을 v3.9에서 v3.11로 변경후 해결되었습니다.

```text
[ERROR] AttributeError: 'kendra' object has no attribute 'retrieve'
Traceback (most recent call last):
  File "/var/task/lambda_function.py", line 215, in lambda_handler
    answer = get_answer_using_template(text)
  File "/var/task/lambda_function.py", line 148, in get_answer_using_template
    relevant_documents = retriever.get_relevant_documents(query)
  File "/var/lang/lib/python3.8/site-packages/langchain/schema/retriever.py", line 208, in get_relevant_documents
    raise e
  File "/var/lang/lib/python3.8/site-packages/langchain/schema/retriever.py", line 201, in get_relevant_documents
    result = self._get_relevant_documents(
  File "/var/lang/lib/python3.8/site-packages/langchain/retrievers/kendra.py", line 421, in _get_relevant_documents
    result_items = self._kendra_query(query)
  File "/var/lang/lib/python3.8/site-packages/langchain/retrievers/kendra.py", line 390, in _kendra_query
    response = self.client.retrieve(**kendra_kwargs)
  File "/var/runtime/botocore/client.py", line 876, in __getattr__
    raise AttributeError(
```
