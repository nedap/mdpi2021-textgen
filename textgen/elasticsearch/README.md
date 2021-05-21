## Example Debug Queries 
To assert that everything works

```curl
curl -XGET "http://localhost:9200/ehr-data-index/_mapping" | jq
```

and

```curl
curl -XGET "http://localhost:9200/ehr-data-index/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "match": {
      "body": {
        "query": "<your query here>"
      }
    }
  }
}' | jq
```
