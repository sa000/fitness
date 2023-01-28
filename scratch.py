
import boto3

# Connect to DynamoDB
dynamodb = boto3.client('dynamodb')

# Table name and column names
table_name = 'my_table'
column_names = ['column1', 'column2', 'column3']

# Create the table
dynamodb.create_table(
    TableName=table_name,
    KeySchema=[
        {
            'AttributeName': column_names[0],
            'KeyType': 'HASH'
        },
    ],
    AttributeDefinitions=[
        {
            'AttributeName': column_name,
            'AttributeType': 'S' # Use 'S' for string, 'N' for number, etc.
        }
        for column_name in column_names
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 1,
        'WriteCapacityUnits': 1
    }
)