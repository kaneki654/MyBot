from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# Mock model inference
def mock_generate(prompt):
    return f"Model responds to: {prompt} with some generated text."

class LightWeightAPI(BaseHTTPRequestHandler):
    def do_POST(self):
        # 1. Read the request length
        content_length = int(self.headers['Content-Length'])
        
        # 2. Read the actual request body
        post_data = self.rfile.read(content_length)
        
        try:
            # 3. Parse JSON
            request_json = json.loads(post_data.decode('utf-8'))
            prompt = request_json.get('prompt', '')
            
            # 4. Generate Text
            output_text = mock_generate(prompt)
            
            # 5. Format Response
            response_data = {'completion': output_text}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            
        except Exception as e:
             self.send_response(400)
             self.end_headers()
             self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, LightWeightAPI)
    print(f"Starting lightweight AI API on port {port}...")
    # httpd.serve_forever() # Uncomment to actually run the server
    print("Server ready to accept requests.")

if __name__ == '__main__':
    run_server()
