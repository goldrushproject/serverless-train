import unittest
import json
import base64
import pickle
from app.app import lambda_handler

class TestApp(unittest.TestCase):
    def test_app(self):
        with open('./tests/sample_data.json') as f:
            test_event = json.load(f)
        context = {}
        response = lambda_handler(test_event, context)
        response_body = json.loads(response['body'])
        
        model_serialized = response_body.pop('model', None)
        print(response_body)
        
        self.assertEqual(response["statusCode"], 200)
        
        # Extract and write the model to a file
        model = pickle.loads(base64.b64decode(model_serialized))
        with open('trained_model.pkl', 'wb') as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    unittest.main()
