import json
import requests
import logging
import boto3

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info("Lambda function invoked.")

    try:
        if isinstance(event, str):
            logger.warning("Event was a string, converting with json.loads")
            event = json.loads(event)

        logger.info(f"Parsed event: {event}")

        # body_raw = event.get('body')
        body = json.loads(event) if isinstance(event, str) else event

        logger.info(f"Parsed request body: {body}")

        texts = body.get("texts")
        model_name = body.get("model_name", "e5_mistral_embed_384")
        logger.info(f"Received texts: {texts} with model: {model_name}")

        if not model_name or not texts:
            logger.warning("Missing required parameters: 'model_name' or 'texts'")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'model_name' or 'texts'"})
            }

        # Setup model endpoint
        base_url = "http://ray.ml.uat.us.workloads.elizacloud.com/serve/"
        if model_name == "e5_mistral_embed_384":
            url = base_url + "e5embed384"
        else:
            logger.error(f"Unsupported model name received: {model_name}")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Unsupported model name: {model_name}"})
            }

        headers = {"Content-Type": "application/json"}
        data = {"texts": texts}
        logger.info(f"Sending POST request to {url} with data: {data}")

        # Make the request
        response = requests.post(url, headers=headers, json=data, verify=False)
        logger.info(f"Received response with status code: {response.status_code}")
        response.raise_for_status()

        response_json = response.json()
        logger.info(f"Raw API response: {response_json}")

        # Extract embeddings
        if "dense_embeddings" in response_json:
            embeddings = response_json["dense_embeddings"]
            logger.info("Embeddings found at top level of response.")
        elif "result" in response_json:
            result = response_json["result"]
            if isinstance(result, str):
                logger.info("Parsing stringified 'result' field into JSON.")
                result = json.loads(result)
            embeddings = result.get("dense_embeddings")
            if embeddings is None:
                logger.error("Missing 'dense_embeddings' in result payload.")
                raise ValueError("Missing 'dense_embeddings' in result")
            logger.info("Embeddings extracted from nested result field.")
        else:
            logger.error("Unexpected response format from model server.")
            raise ValueError("Unexpected response format")

        logger.info("Returning embeddings successfully.")
        return {
            "statusCode": 200,
            "body": json.dumps({"embeddings": embeddings})
        }

    except requests.exceptions.RequestException as e:
        logger.exception("Request to embedding model failed.")
        return {
            "statusCode": 502,
            "body": json.dumps({"error": f"Request error: {str(e)}"})
        }
    except Exception as e:
        logger.exception("Unhandled exception occurred during processing.")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Internal server error: {str(e)}"})
        }
