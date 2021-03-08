import re
import json
from urllib.parse import urlparse, parse_qs
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
# from urllib.parse import urlparse, parse_qs


class ServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        path = self.path
        if path == "/":
            path = "/index"

        try:
            file  = open("pages"+path + ".html", 'r')
        except FileNotFoundError:
            file  = open("pages/404.html", 'r')

        message = file.read()
        file.close()
        self.wfile.write(bytes(message, "utf8"))
        return

    def do_POST(self):



        path = self.path
        #Обработчик подписки
        if path == "/email":
            self.send_response(301)
            self.send_header('Location', '/support')
            self.end_headers()
            content_len = int(self.headers.get('Content-Length'))
            post_data = self.rfile.read(content_len)
            logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                         str(self.path), str(self.headers), post_data.decode('utf-8'))

            email = re.split(r"email=",str(post_data))[1]
            email = re.sub(r"\'","",email)
            print(email)


        if path == "/position":
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            content_len = int(self.headers.get('Content-Length'))
            post_data = self.rfile.read(content_len)
            parsed = post_data.decode('utf-8')
            json_data = json.loads(parsed)

            print(json_data)
            print(type(json_data))
            logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                         str(self.path), str(self.headers), post_data.decode('utf-8'))

            self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

        return


if __name__ == '__main__':
    # server = HTTPServer(('192.168.50.200', 8081), ServerHandler)
    server = HTTPServer(('127.0.0.1', 8081), ServerHandler)
    server.serve_forever()
