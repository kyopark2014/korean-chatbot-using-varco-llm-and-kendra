# CDK 인프라 배포하기

여기서는 [cdk-varco-ko-llm-stack.ts](./cdk-varco-ko-llm/cdk-varco-ko-llm-stack.ts)에서 정의한 인프라 코드에 대하여 설명합니다.


관련 환경 변수는 아래와 같습니다. enpoint 이름은 VARCO LLM을 SageMaker로 설치할때 생성된 "endpoint name"입니다. 필요시 varco_region을 업데이트 합니다.

```java
const s3_prefix = 'docs';
const projectName = `chatbot-varco-kendra-${region}`;
const bucketName = `storage-for-${projectName}`;
const endpoint_name = 'endpoint-varco-llm-ko-13b-ist-1';
const varico_region =  "us-west-2"; // "us-west-2"
```

Chatbot UI를 보여지기 위하여 CloudFront는 S3에 저장된 html, css, image파일을 로드합니다. 또한 S3는 Kendra에 문서 파일을 업로드하기 스토리지로도 사용됩니다.

```java
const s3Bucket = new s3.Bucket(this, `storage-${projectName}`, {
    bucketName: bucketName,
    blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
    autoDeleteObjects: true,
    publicReadAccess: false,
    versioned: false,
    cors: [
        {
            allowedHeaders: ['*'],
            allowedMethods: [
                s3.HttpMethods.POST,
                s3.HttpMethods.PUT,
            ],
            allowedOrigins: ['*'],
        },
    ],
});
```

Call log를 저장하고 활용하기 위하여, 아래와 같이  DyanmoDB를 정의합니다.

```java
const callLogTableName = `db-call-log-for-${projectName}`;
const callLogDataTable = new dynamodb.Table(this, `db-call-log-for-${projectName}`, {
    tableName: callLogTableName,
    partitionKey: { name: 'user-id', type: dynamodb.AttributeType.STRING },
    sortKey: { name: 'request-id', type: dynamodb.AttributeType.STRING },
    billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
});
const callLogIndexName = `index-type-for-${projectName}`;
callLogDataTable.addGlobalSecondaryIndex({ // GSI
    indexName: callLogIndexName,
    partitionKey: { name: 'type', type: dynamodb.AttributeType.STRING },
});
```



