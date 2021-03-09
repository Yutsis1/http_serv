import json
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
from typing import Tuple


class S(BaseHTTPRequestHandler):
    _use_text = False

    def _set_response(self):
        self.send_response(200)
        if self._use_text:
            self.send_header('Content-type', 'text/html')
        else:
            self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        if self._use_text:
            self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))
        else:
            dict_for_answer = {"method": "GET", "content": {"lol": "kek"}}
            json_str = json.dumps(dict_for_answer)
            self.wfile.write(json_str.encode(encoding='utf_8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                     str(self.path), str(self.headers), post_data.decode('utf-8'))

        self._set_response()
        if self._use_text:
            self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))
        else:
            dict_for_answer = {"method": "POST", "content": {"lol": "kek"}}
            json_str = json.dumps(dict_for_answer)
            self.wfile.write(json_str.encode(encoding='utf_8'))


def run(server_class=HTTPServer, handler_class=S, port=8081, address='127.0.0.1'):
    logging.basicConfig(level=logging.INFO)
    server_address = (address, port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')


if __name__ == '__main__':
    # from sys import argv
    #
    # if len(argv) == 2:
    #     run(port=int(argv[1]), address='192.168.50.200')
    # else:
    #     run()
    run(address='192.168.50.200')
    # run()
