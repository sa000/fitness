import boto3
from boto3.dynamodb.conditions import Key, Attr

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("feature_store")
a = table.scan(TableName = "feature_store",
    FilterExpression=Attr("excercise").eq("squat")
    )
print(a)