import unittest
import json
from app.app import lambda_handler

class TestApp(unittest.TestCase):
    def test_app(self):
        with open('./tests/assessment_test_event.json') as f:
            test_event = json.load(f)
        context = {}
        response = lambda_handler(test_event, context)
        response_body = json.loads(response['body'])
        
        print("Predicted Prices:", response_body['predicted_prices'])
        
        self.assertEqual(response["statusCode"], 200)
        self.assertIn("Hello from Lambda!", response_body["message"])

if __name__ == "__main__":
    unittest.main()
