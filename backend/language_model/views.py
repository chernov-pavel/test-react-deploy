from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.utils import json

from neural_network.configuration import SEQUENCE_LENGTH
from neural_network.main import NeuralNetwork


@api_view(['POST'])
def get_continue_by_input(request):
    neural_network = NeuralNetwork.get_instance()
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    text = body['text']
    if len(text) == 0:
        return Response([])
    start_index = len(text)-SEQUENCE_LENGTH if len(text) > SEQUENCE_LENGTH else 0
    result = list(neural_network.predict_completions(text[start_index:].lower(), 5))
    return Response(result)
