# VARCO LLM와 Amazon Kendra를 이용하여 한국어 Chatbot 만들기

여기서는 [VARCO LLM](https://ncsoft.github.io/ncresearch/varco-llm/)와 [Amazon Kendra](https://docs.aws.amazon.com/ko_kr/kendra/latest/dg/what-is-kendra.html)를 이용하여 Question/Answering을 위한 한국어 Chatbot을 구현하고자 합니다. VARCO LLM은 엔씨소프트(NC SOFT)에서 제공하는 대용량 언어 모델(LLM)입니다. VARCO LLM KO-13B-IST는 VARCO LLM KO-13B-FM의 파인튜닝 모델로서 Question and Answering, Summarization등 다양한 태스크에 활용할 수 있으며, [Amazon SageMaker](https://aws.amazon.com/marketplace/seller-profile?id=seller-tkuvdeznmi2w4)를 이용하여 쉽게 배포하여 사용할 수 있습니다.  

대규모 언어 모델(LLM)을 학습할 때에 포함되지 못한 특정 영역의 데이터(domain specific data)에 대해 질문을 하면 답변할 수 없습니다. [RAG (Retrieval Augmented Generation) ](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html)를 이용하면, 외부 문서 저장소에서 질문(Question)에 관련된 문서를 읽어와서 Prompt에 활용하는 방식으로 대용량 모델의 성능을 강화될 수 있습니다. Amazon Kendra는 자연어 처리 및 고급 기계 학습 알고리즘을 사용하여 데이터에서 검색 질문에 대한 답변을 얻는 지능형 검색 서비스로서 대규모 언어 모델에 RAG를 구현할 때 유용하게 활용될 수 있습니다. 

여기서는 대규모 언어 모델을 위한 어플리케이션 개발 프레임워크인 [LangChain](https://www.langchain.com/)을 활용하여 어플리케이션을 개발하며, Amazon의 대표적인 [서버리스 서비스](https://aws.amazon.com/ko/serverless/)인 [Amazon Lambda](https://aws.amazon.com/ko/lambda/)로 서빙하는 인프라를 구축합니다. Amazon Lambda를 비롯한 인프라를 배포하고 관리하기 위하여 [Amazon CDK](https://aws.amazon.com/ko/cdk/)를 활용합니다.

전체적인 Architecture는 아래와 같습니다. 사용자의 질문은 Query로 [Amazon CloudFront](https://aws.amazon.com/ko/cloudfront/)와 [Amazon API Gateway](https://aws.amazon.com/ko/api-gateway/)를 거쳐서, Lambda에 전달됩니다. Lambda는 Kendra로 Query를 전달하여 관련된 문서들의 발췌를 받은후에 VARCO LLM에 전달하여 답변을 얻습니다. 이후 답변은 사용자에게 전달되어 채팅화면에 표시됩니다. 또한 채팅이력은 [Amazon DynamoDB](https://aws.amazon.com/ko/dynamodb/)를 이용해 저장되고 활용됩니다.

<img src="https://github.com/kyopark2014/korean-chatbot-using-varco-llm-and-kendra/assets/52392004/b0d38264-3a65-4c89-8477-cb8f291b6ebf" width="800">


## LangChain과 연동하기 

VARCO LLM의 Input형태는 아래와 같습니다.

```java
{
  "text": "input text here",
  "request_output_len": 512,
  "repetition_penalty": 1.1,
  "temperature": 0.1,
  "top_k": 50,
  "top_p": 0.9
}
```
VARCO LLM의 Output의 기본 포맷은 아래와 같습니다.

```java
{
  "result": [
    "output text here"
  ]
}
```

LangChain은 LLM application의 개발을 도와주는 Framework으로 Question anc Answering, Summarization등 다양한 task에 맞게 Chain등을 활용하여 편리하게 개발할 수 있습니다. VARCO LLM은 SageMaker Endpoint로 배포되므로 아래와 같이 VARCO LLM의 입력과 출력의 포맷을 맞추어서 ContentHandler를 정의합니다. 상세한 내용은 [lambda-chat](./lambda-chat/lambda_function.py)에서 확인할 수 있습니다.

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



### 문서를 Kendra에 올리기

여기서는 Kendra에 문서를 등록할때에 S3를 이용합니다. [lambda-upload](./lambda-upload/index.js)에서는 대용량 파일을 쉽게 S3에 올릴수 있도록 [getSignedUrlPromise()](https://docs.aws.amazon.com/AWSJavaScriptSDK/latest/AWS/S3.html)을 이용하여 [presidgned url](https://docs.aws.amazon.com/AmazonS3/latest/userguide/PresignedUrlUploadObject.html)을 생성합니다.

```java
const URL_EXPIRATION_SECONDS = 300;
const s3Params = {
    Bucket: bucketName,
    Key: s3_prefix + '/' + filename,
    Expires: URL_EXPIRATION_SECONDS,
    ContentType: contentType,
};

const uploadURL = await s3.getSignedUrlPromise('putObject', s3Params);
```

이후 client가 presigned url을 이용하여 문서 파일을 업로드한 후에, 아래와 같이 [lambda-chat](./lambda-chat/lambda_function.py)은 [batch_put_document](https://docs.aws.amazon.com/ko_kr/kendra/latest/dg/in-adding-binary-doc.html)를 이용하여 Kendra에 등록을 요청합니다.

```python
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
```

### Query와 관련된 문서의 발췌를 Kendra로 부터 가져오기 

Kendra를 LangChain의 Retreiver로 설정합니다.

```python
from langchain.retrievers import AmazonKendraRetriever
retriever = AmazonKendraRetriever(index_id=kendraIndex)
```

LangChain의 [RetrievalQA](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html?highlight=retrievalqa#langchain.chains.retrieval_qa.base.RetrievalQA)와 Kendra Retriever를 이용하여 Query와 관련된 문서를 읽어옵니다.

```python
prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

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
```

Kendra로 부터 가져온 관련된 문서의 meta data로 부터 reference에 대한 정보를 아래처럼 추출할 수 있습니다.

```python
source_documents = result['source_documents']        
reference = get_reference(source_documents)

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
```

### VARCO LLM에 Query 하기

Kendra의 [Characters in query text](https://us-west-2.console.aws.amazon.com/servicequotas/home/services/kendra/quotas/L-7107C1BC)에 따라, 1000 이하(기본값)인 경우에 Kendra로 관련된 문서를 조회할 수 있습니다. 이 숫자는 어플리케이션의 용도에 따라 Quota 변경을 AWS에 요청할 수 있습니다. 아래에서는 1000(기본값) 이하의 질문에 대해 질의를 수행하고 있습니다.

```python
querySize = len(text)
print('query size: ', querySize)

if querySize<1000: 
    answer = get_answer_using_template(text)
else:
    answer = llm(text)      
```

### VARCO LLM의 결과 전달하기 

VARCO LLM의 응답에서 "### Assistant:" 이하룰 추출하여 사용자에게 메시지의 형태로 전달합니다.

```python
pos = answer.rfind('### Assistant:\n') + 15
msg = answer[pos:]
```

### AWS CDK로 인프라 구현하기

[CDK 구현 코드](./cdk-varco-ko-llm/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다.

## 직접 실습 해보기

### 사전 준비 사항

1) 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)

2) VARCO LLM을 위하여, "ml.g5.12xlarge"를 사용합니다. [Service Quotas - AWS services - Amazon SageMaker](https://us-west-2.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas)에서 "ml.g5.12xlarge for endpoint usage"를 최소 1개 이상으로 할당 받아야 합니다. 만약 Quota가 0인 경우에 [Request quota increase]을 선택하여 요청합니다.


### CDK를 이용한 인프라 설치
[인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 


### 실행결과

VARCO LLM 학습시에 Amazon Kendra관련 데이터가 포함되지 않았으므로 아래와 같이 Kendra에 대한 질문을 할 경우에 기대하지 않은 답변을 합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-varco-llm-and-kendra/assets/52392004/fc30dfec-2256-450e-aa83-f9d61b354e22)

Kendra와 관련된 문서인 [Amazon_Kendra.pdf](./Amazon_Kendra.pdf)를 다운받은 후에 채팅창의 파일버튼을 선택하여 업로드하면, Kendra에 문서가 등록되고 아래와 같이 문서의 요약(Summarization)을 확인할 수 있습니다. Amazon_Kendra.pdf는 Kendra 서비스에 대한 소개자료입니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-varco-llm-and-kendra/assets/52392004/d95498be-3bff-4c95-b065-20deb6860d99)

이후 동일한 질문을 다시하면 기대했던 Kendra에 대한 정보를 확인할 수 있습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-varco-llm-and-kendra/assets/52392004/d92d151c-b4a3-41e1-b6ef-7a51dc3f9d8d)


### 리소스 정리하기

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. [Cloud9 console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)에 접속하여 아래와 같이 삭제를 합니다.

```java
cdk destroy
```

본 실습에서는 VARCO LLM의 endpoint로 "ml.g5.12xlarge"를 사용하고 있으므로, 더이상 사용하지 않을 경우에 반드시 삭제하여야 합니다. 특히 cdk destroy 명령어로 Chatbot만 삭제할 경우에 SageMaker Endpoint가 유지되어 지속적으로 비용이 발생될 수 있습니다. 이를 위해 [Endpoint Console](https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/endpoints)에 접속해서 Endpoint를 삭제합니다. 마찬가지로 [Models](https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/models)과 [Endpoint configuration](https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/endpointConfig)에서 설치한 VARCO LLM의 Model과 Configuration을 삭제합니다.

## 결론

엔씨소프트의 한국어 언어모델인 VARCO LLM과 Amazon Kendra를 활용하여 질문과 답변(Question/Answering) 테스크를 수행하는 Chatbot 어플리케이션을 구현하였습니다. 대규모 언어 모델(LLM)을 활용하면 기존 Rule 기반의 Chatbot보다 훨씬 강화된 기능을 제공할 수 있습니다. 대규모 언어모델 확습에 포함되지 못한 특정 영역의 데이터는 Amazon Kendra를 통해 보완될수 있으며, 이를 통해 엔터프라이즈 기업과 같이 질문과 답변을 고객에게 제공하는 기업들에 유용하게 사용될 수 있을것으로 보여집니다. 또한 대규모 언어 모델을 개발하는 프레임워크인 LangChain을 VARCO LLM과 연동하는 방법과 Amazon Kendra와 관련된 서빙 인프라를 AWS CDK를 활용하여 쉽게 구현할 수 있었습니다. 한국어 대규모 언어 모델은 Chatbot뿐 아니라 향후 다양한 분야에서 유용하게 활용될 수 있을것으로 기대됩니다.

## Reference

[NC - github](https://ncsoft.github.io/ncresearch/varco-llm/)

[Deploy VARCO LLM Model 13B IST Package from AWS Marketplace](https://github.com/ncsoft/ncresearch/blob/main/notebooks/varco_model_13_IST.ipynb)

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
