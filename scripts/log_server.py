from flask import Flask, request
import logging

app = Flask(__name__)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route('/log', methods=['POST'])
def receive_log():
    # Get log message from request
    log_data = request.get_json() if request.is_json else request.form.to_dict()
    
    # Print the log message
    print(f"Received log: {log_data}")
    
    return {"status": "success", "message": "Log received"}

@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "up"}

if __name__ == '__main__':
    import argparse
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Flask Log Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    
    args = parser.parse_args()
    
    print(f"Starting log server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)
