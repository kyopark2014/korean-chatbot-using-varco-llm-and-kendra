# Varco LLM을 이용하여 간단한 한국어 Chatbot 구현하기

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
