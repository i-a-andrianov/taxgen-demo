curl -i \
     -H "Accept: application/json" \
     -H "Content-Type: application/json" \
     -X GET \
     http://172.17.0.2:8888/current?uid=0d67be40-79e6-4523-8988-8dd7b2499617



curl -X POST http://172.17.0.2:8888/generate/words \
     -H "Content-Type: application/json" \
     -d '{
           "uid": "0d67be40-79e6-4523-8988-8dd7b2499617",
           "start_node": "maltese_dog.n.01"
         }'

